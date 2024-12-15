import threading
import time
import random
from queue import Queue
from tqdm import tqdm
import json
import os
import random

class MultiProcessor:
    def __init__(self, llm, parse_method, data_template, prompt_template, correction_template, 
                 validator, time_limit=120, back_up_llm=None, temperature=0.7, IsPromptList=False, checkpoint_dir=None):
        self.llm = llm
        self.back_up_llm = back_up_llm
        self.parse_method = parse_method
        self.data_template = data_template
        self.prompt_template = prompt_template
        self.correction_template = correction_template
        self.validator = validator
        self.time_limit = time_limit
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else 'checkpoint'
        self.temperature = temperature
        self.IsPromptList = IsPromptList
        self.checkpoint_path_0 = os.path.join(self.checkpoint_dir, 'checkpoint_0.json')
        self.checkpoint_path_1 = os.path.join(self.checkpoint_dir, 'checkpoint_1.json')
        self.choose_checkpoint = self.initialize_checkpoint_choice()

    def initialize_checkpoint_choice(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        if not os.path.exists(self.checkpoint_path_0):
            with open(self.checkpoint_path_0, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.checkpoint_path_1):
            with open(self.checkpoint_path_1, 'w') as f:
                json.dump({}, f)
        
        size_0 = os.path.getsize(self.checkpoint_path_0)
        size_1 = os.path.getsize(self.checkpoint_path_1)
        return size_1 >= size_0


    def generate_prompt(self, **kwargs):
        kwargs['data_template'] = self.data_template
        if self.IsPromptList:
            selected_template = random.choice(self.prompt_template)
            prompt = selected_template.format(**kwargs)
        else:
            prompt = self.prompt_template.format(**kwargs)
        return prompt

    def generate_correction_prompt(self, answer):
        return self.correction_template.format(answer=answer, data_template=self.data_template)

    def task_perform(self, llm, **kwargs):
        try:
            prompt = self.generate_prompt(**kwargs)

            answer = llm.ask(prompt, self.temperature)
            structured_data = self.parse_method(answer)
            return structured_data
        except Exception as e:
            print(f"Error in task_perform: {str(e)}")
            raise e

    def correct_data(self, llm, answer):
        correction_prompt = self.generate_correction_prompt(answer)
        correction = llm.ask(correction_prompt)
        return self.parse_method(correction)

    def process_task(self, index, key_dict, Active_Transform):
        try:
            attempts = 0
            base_wait_time = 1
            use_backup = False

            while attempts < 2:
                try:
                    current_llm = self.back_up_llm if (use_backup and self.back_up_llm is not None) else self.llm
                    structured_data = self.task_perform(current_llm, **key_dict)
                    if self.validator(structured_data):
                        return self.map_answer_to_pos(structured_data) if Active_Transform else structured_data
                    corrected_answer = self.correct_data(current_llm, structured_data)
                    if corrected_answer and self.validator(corrected_answer):
                        return self.map_answer_to_pos(corrected_answer) if Active_Transform else corrected_answer
                    break
                except Exception as e:
                    if 'Throttling.RateQuota' in str(e):
                        wait_time = base_wait_time * (2 ** attempts) + random.uniform(0, 1)
                        print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds. Attempt {attempts + 1}/2")
                        time.sleep(wait_time)
                    else:
                        print(f"An error occurred: {str(e)}. Attempt {attempts + 1}/2")
                    
                    attempts += 1
                    if not use_backup and self.back_up_llm is not None:
                        use_backup = True
                        print(f"Switching to backup LLM to process task {index}")

            return None

        except Exception as final_error:
            print(f"Error occurred during process_task for index {index}: {str(final_error)}")
            return None

    def map_answer_to_pos(self, answer_dict):
        pos_dict = {}
        for i, (key, value) in enumerate(answer_dict.items(), start=1):
            pos_key = f"pos{i}"
            pos_dict[pos_key] = value
        return pos_dict

    def save_checkpoint(self, results, checkpoint_type='task'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        existing_results = self.load_checkpoint(checkpoint_type)
        
        results_copy = results.copy()
        
        for k, v in results_copy.items():
            if v is not None:
                existing_results[str(k)] = v

        if checkpoint_type == 'similarity':
            checkpoint_path_0 = os.path.join(self.checkpoint_dir, 'similarity_checkpoint_0.json')
            checkpoint_path_1 = os.path.join(self.checkpoint_dir, 'similarity_checkpoint_1.json')
        else:  # 'task' or any other type
            checkpoint_path_0 = self.checkpoint_path_0
            checkpoint_path_1 = self.checkpoint_path_1

        if self.choose_checkpoint:
            checkpoint_path = checkpoint_path_0
            self.choose_checkpoint = False
        else:
            checkpoint_path = checkpoint_path_1
            self.choose_checkpoint = True
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=4)
        print(f"{checkpoint_type.capitalize()} checkpoint updated and saved at {checkpoint_path}.")

    def load_checkpoint(self, checkpoint_type='task'):
        if checkpoint_type == 'similarity':
            checkpoint_path_0 = os.path.join(self.checkpoint_dir, 'similarity_checkpoint_0.json')
            checkpoint_path_1 = os.path.join(self.checkpoint_dir, 'similarity_checkpoint_1.json')
        else:  # 'task' or any other type
            checkpoint_path_0 = self.checkpoint_path_0
            checkpoint_path_1 = self.checkpoint_path_1

        checkpoint_path = checkpoint_path_1 if self.choose_checkpoint else checkpoint_path_0

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Warning: No {checkpoint_type} checkpoint found at {checkpoint_path}. Starting from scratch.")
            return {}

    def calculate_similarities(self, index_dict, key1, key2, num_threads, checkpoint_dir=None, checkpoint=10, active_reload=False):
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        
        # 确保checkpoint目录存在
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if active_reload:
            previous_results = self.load_checkpoint('similarity')
            results = {k: v for k, v in previous_results.items()}
            remaining_tasks = {k: v for k, v in index_dict.items() if k not in results}
            print(f"原始数据数量: {len(index_dict)}")
            print(f"已完成任务数量: {len(results)}")
            print(f"剩余待处理数据数量: {len(remaining_tasks)}")
        else:
            results = {}
            remaining_tasks = index_dict

        queue = Queue()
        checkpoint_counter = 0

        for index, key_dict in remaining_tasks.items():
            queue.put((index, key_dict))

        def worker(pbar):
            nonlocal checkpoint_counter
            while not queue.empty():
                index, key_dict = queue.get()
                if index in results:
                    queue.task_done()
                    pbar.update(1)
                    continue

                try:
                    text1 = key_dict.get(key1, "")
                    text2 = key_dict.get(key2, "")
                    similarity = self.llm.calculate_similarity(text1, text2)
                    results[index] = similarity
                except Exception as e:
                    print(f"Error calculating similarity for index {index}: {str(e)}")
                    results[index] = None

                queue.task_done()
                pbar.update(1)

                checkpoint_counter += 1
                if checkpoint_counter % checkpoint == 0:
                    print(f"Saving checkpoint at counter {checkpoint_counter}")
                    self.save_checkpoint(results, 'similarity')

        with tqdm(total=len(remaining_tasks)) as pbar:
            threads = []
            for _ in range(min(num_threads, len(remaining_tasks))):
                thread = threading.Thread(target=worker, args=(pbar,))
                threads.append(thread)
                thread.start()

            queue.join()

            for thread in threads:
                thread.join()

        print("Final save_checkpoint call")
        self.save_checkpoint(results, 'similarity')

        completed_tasks = sum(1 for r in results.values() if r is not None)
        print(f"Final results - Total tasks: {len(results)}, Completed tasks: {completed_tasks}")

        return results

    def multitask_perform(self, index_dict, num_threads, checkpoint=10, Active_Reload=False, Active_Transform=False, checkpoint_dir=None):
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_path_0 = os.path.join(self.checkpoint_dir, 'checkpoint_0.json')
            self.checkpoint_path_1 = os.path.join(self.checkpoint_dir, 'checkpoint_1.json')
            self.choose_checkpoint = self.initialize_checkpoint_choice()

        if Active_Reload:
            previous_results = self.load_checkpoint()
            
            results = {k: v for k, v in previous_results.items()}
            index_dict = {k: v for k, v in index_dict.items()}
            
            completed_tasks = {k: v for k, v in results.items() if v is not None}
            
            remaining_tasks = {k: v for k, v in index_dict.items() if k not in completed_tasks}
            
            print(f"原始数据数量: {len(index_dict)}")
            print(f"已完成任务数量: {len(completed_tasks)}")
            print(f"剩余待处理数据数量: {len(remaining_tasks)}")
        else:
            results = {}
            remaining_tasks = index_dict

        queue = Queue()
        checkpoint_counter = 0

        for index, key_dict in remaining_tasks.items():
            queue.put((index, key_dict))

        def worker(pbar):
            nonlocal checkpoint_counter
            while not queue.empty():
                index, key_dict = queue.get()
                if index in results and results[index] is not None:
                    queue.task_done()
                    pbar.update(1)
                    continue

                result = None
                thread_result_queue = Queue()

                thread = threading.Thread(target=lambda q, arg1, arg2, arg3: q.put(self.process_task(arg1, arg2, arg3)), 
                                          args=(thread_result_queue, index, key_dict, Active_Transform))
                thread.start()
                thread.join(timeout=self.time_limit)

                if thread.is_alive():
                    print(f"Thread processing task {index} timed out.")
                    thread.join()
                else:
                    if not thread_result_queue.empty():
                        result = thread_result_queue.get()
                    else:
                        print(f"No result obtained for task {index}")

                if result is not None:
                    results[index] = result

                queue.task_done()
                pbar.update(1)

                checkpoint_counter += 1
                if checkpoint_counter % checkpoint == 0:
                    print(f"Saving checkpoint at counter {checkpoint_counter}")
                    self.save_checkpoint(results)

        with tqdm(total=len(remaining_tasks)) as pbar:
            threads = []
            for _ in range(min(num_threads, len(remaining_tasks))):
                thread = threading.Thread(target=worker, args=(pbar,))
                threads.append(thread)
                thread.start()

            queue.join()

            for thread in threads:
                thread.join()

        print("Final save_checkpoint call")
        self.save_checkpoint(results)

        completed_tasks = sum(1 for r in results.values() if r is not None)
        print(f"Final results - Total tasks: {len(results)}, Completed tasks: {completed_tasks}")

        return results
    
    def multitask_manage(self, index_dict, num_threads, checkpoint=10, Active_Reload=False, Active_Transform=False, checkpoint_dir=None, 
                        threshold=None, max_multitask_retries=None, max_time=None):
        start_time = time.time()
        total_tasks = len(index_dict)
        retry_count = 0
        results = {}

        while True:
            # 检查时间限制
            if max_time and (time.time() - start_time) >= max_time:
                print("Reached maximum time limit.")
                break

            # 检查重试次数限制
            if max_multitask_retries and retry_count >= max_multitask_retries:
                print("Reached maximum retry limit.")
                break

            # 执行multitask_perform
            if retry_count == 0:
                # 首次执行，使用传入的Active_Reload
                results = self.multitask_perform(index_dict, num_threads, checkpoint, Active_Reload, Active_Transform, checkpoint_dir)
            else:
                # 重试时，强制Active_Reload=True
                results = self.multitask_perform(index_dict, num_threads, checkpoint, True, Active_Transform, checkpoint_dir)

            # 计算完成任务的比例
            completed_tasks = sum(1 for r in results.values() if r is not None)
            completion_ratio = completed_tasks / total_tasks

            print(f"Completed {completed_tasks}/{total_tasks} tasks. Completion ratio: {completion_ratio:.2f}")

            # 检查是否达到阈值
            if threshold is None or completion_ratio >= threshold:
                print("Reached or exceeded completion threshold.")
                break

            retry_count += 1
            print(f"Starting retry {retry_count}")

        return results
