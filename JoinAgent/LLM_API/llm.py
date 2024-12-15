import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import base64
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time
import random
from collections import deque
import warnings

class ModelPool:
    def __init__(self, models, weights=None):
        """
        初始化模型池
        :param models: 一个包含 MultiLLM 或 DeepSeekLLM 实例的列表
        :param weights: 一个权重列表，如果为 None 或长度不足，则自动补齐
        """
        self.models = []
        self.weights = []
        self.model_types = {}
        self.available_operations = set()
        
        if weights is None:
            weights = [1] * len(models)
        
        if len(weights) < len(models):
            avg_weight = sum(weights) / len(weights)
            weights.extend([avg_weight] * (len(models) - len(weights)))
            warnings.warn(f"Number of weights less than number of models. Extended weights list with average value: {avg_weight:.2f}")
        elif len(weights) > len(models):
            weights = weights[:len(models)]
            warnings.warn(f"Number of weights greater than number of models. Using only the first {len(models)} weights.")

        for model, weight in zip(models, weights):
            self.add_model(model, weight_factor=weight)

    def merge_models(self, model):
        """
        检查是否可以合并模型，如果可以则合并
        :param model: 要检查的模型
        :return: 如果模型被合并返回 True，否则返回 False
        """
        if isinstance(model, DeepSeekLLM):
            for existing_model in self.models:
                if isinstance(existing_model, DeepSeekLLM) and existing_model.api_key == model.api_key:
                    print(f"Merging DeepSeekLLM model: Keeping existing model with API key {existing_model.api_key}")
                    return True
        elif isinstance(model, MultiLLM):
            for existing_model in self.models:
                if (isinstance(existing_model, MultiLLM) and 
                    existing_model.api_key == model.api_key and 
                    existing_model.model == model.model):
                    print(f"Merging MultiLLM model: Keeping existing model {existing_model.model} with API key {existing_model.api_key}")
                    return True
        return False

    def add_model(self, model, weight_factor=1):
        """
        向模型池中添加一个新模型
        :param model: 要添加的模型
        :param weight_factor: 权重因子，用于调整默认权重，默认为1
        """
        if self.merge_models(model):
            return  # 如果模型被合并，不需要继续添加

        model_type = type(model).__name__
        model_operations = set(model.show_type())
        
        if self.weights:
            avg_weight = sum(self.weights) / len(self.weights)
        else:
            avg_weight = 1

        new_weight = avg_weight * weight_factor

        self.models.append(model)
        self.weights.append(new_weight)
        self.model_types[model] = model_operations
        self.available_operations.update(model_operations)
        self.normalize_weights()

    def normalize_weights(self):
        """
        归一化权重，使其总和为1
        """
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def remove_model(self, model):
        """
        从模型池中移除一个模型
        """
        if model in self.models:
            index = self.models.index(model)
            self.models.pop(index)
            self.weights.pop(index)
            operations = self.model_types.pop(model) #目前没用，供日志维护
            
            self.update_available_operations()
            self.normalize_weights()
        else:
            print(f"Model not found in the pool")

    def update_available_operations(self):
        """
        更新池中可用的操作列表
        """
        self.available_operations = set()
        for operations in self.model_types.values():
            self.available_operations.update(operations)

    def get_models_for_operation(self, operation):
        """
        获取支持特定操作的模型列表及其权重
        """
        eligible_models = []
        eligible_weights = []
        for model, weight in zip(self.models, self.weights):
            if operation in self.model_types[model]:
                eligible_models.append(model)
                eligible_weights.append(weight)
        return eligible_models, eligible_weights

    def execute_operation(self, operation, *args, **kwargs):
        """
        执行特定操作，根据权重在支持该操作的模型中选择
        """
        if operation not in self.available_operations:
            raise ValueError(f"Operation '{operation}' is not supported by any model in the pool")

        eligible_models, eligible_weights = self.get_models_for_operation(operation)
        if not eligible_models:
            raise ValueError(f"No models available for operation '{operation}'")

        # 根据权重选择一个支持该操作的模型
        chosen_model = random.choices(eligible_models, weights=eligible_weights, k=1)[0]
        
        # 执行操作
        return getattr(chosen_model, operation)(*args, **kwargs)

    def ask(self, prompt, temperature=0.7):
        return self.execute_operation('ask', prompt, temperature=temperature)

    def look(self, image_path, prompt="What's in this image?"):
        return self.execute_operation('look', image_path, prompt=prompt)

    def embed_text(self, input_text):
        return self.execute_operation('embed', input_text)

    def get_pool_size(self):
        """
        返回模型池的大小
        """
        return len(self.models)

    def get_models(self):
        """
        返回模型池中的所有模型及其权重
        """
        return list(zip(self.models, self.weights))

    def get_available_operations(self):
        """
        返回池中所有可用的操作
        """
        return list(self.available_operations)

    def set_weight(self, model, weight_factor):
        """
        调整特定模型的权重
        :param model: 要调整权重的模型
        :param weight_factor: 权重调整因子（倍率）
        """
        if model in self.models:
            index = self.models.index(model)
            self.weights[index] *= weight_factor
            self.normalize_weights()
        else:
            raise ValueError("Model not found in the pool")

    def get_weight(self, model):
        """
        获取特定模型的权重
        """
        if model in self.models:
            index = self.models.index(model)
            return self.weights[index]
        else:
            raise ValueError("Model not found in the pool")

    def update_weights(self, new_weights):
        """
        更新整个权重列表
        :param new_weights: 新的权重列表
        """
        if len(new_weights) < len(self.models):
            avg_weight = sum(new_weights) / len(new_weights)
            new_weights.extend([avg_weight] * (len(self.models) - len(new_weights)))
            warnings.warn(f"Number of weights less than number of models. Extended weights list with average value: {avg_weight:.2f}")
        elif len(new_weights) > len(self.models):
            new_weights = new_weights[:len(self.models)]
            warnings.warn(f"Number of weights greater than number of models. Using only the first {len(self.models)} weights.")
        
        self.weights = new_weights
        self.normalize_weights()


class DeepSeekLLM:
    def __init__(self, version='coder', api_key=None):
        load_dotenv()
        self.version = 'deepseek-' + version
        self.client = None
        self.initialized = False
        self.total_tokens_used = 0

        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('DEEPSEEK_API', None)
        
        base_url = 'https://api.deepseek.com'
        self.init_service(self.api_key, base_url)

    @classmethod
    def show_type(cls):
        return ['ask']

    def init_service(self, api_key: str, base_url: str) -> bool:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.initialized = True
        return True

    def ask(self, prompt: str,temperature=1) -> str:
        if not self.initialized:
            raise ValueError("服务未初始化，请先调用 init_service 方法初始化服务。")

        if not self.client:
            raise ValueError("OpenAI 客户端未正确初始化，请检查初始化过程。")

        response = self.client.chat.completions.create(
            model=self.version,
            messages=[{"role": "user", "content": prompt}]
        )

        if response:
            total_tokens = response.usage.total_tokens
            self.total_tokens_used += total_tokens
            return response.choices[0].message.content
        else:
            return ""

class MultiLLM:
    def __init__(self, model='deepseek-coder', vision_model='gpt-4o-mini', embed_model='text-embedding-3-large', api_key=None):
        load_dotenv()
        self.model = model
        self.vision_model = vision_model
        self.embed_model = embed_model
        
        # 如果初始化时没有提供 api_key，则从环境变量中读取
        if api_key is None:
            self.api_key = os.getenv('MULTI_LLM_API')
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError("API key not found. Please set the MULTI_LLM_API environment variable.")

    @classmethod
    def show_type(cls):
        return ['ask', 'look', 'embed']

    @classmethod
    def test_model(cls, model_types=None, concurrent_threads=200, total_questions=1000, test_prompt="你好,这是第{question_number}次询问", 
                timeout=60, detail=False, policy=None, apply=False):
        available_models = [
            'gpt-4o-2024-08-06',
            'chatgpt-4o-latest',
            'deepseek-coder',
            'glm-4-0520',
            'llama-3.1-405b',
            'claude-3-5-sonnet-20240620'
        ]

        if model_types is None:
            model_types = available_models
        elif isinstance(model_types, str):
            model_types = [model_types]

        valid_models = [model for model in model_types if model in available_models]
        invalid_models = [model for model in model_types if model not in available_models]

        if invalid_models:
            print(f"Warning: The following models are not available: {', '.join(invalid_models)}")

        if not valid_models:
            print("No valid models to test.")
            return

        models_data = []

        for model in valid_models:
            if detail:
                print(f"\nTesting model: {model}")
            else:
                print(f"\nTesting model: {model} - 测试中...")

            llm = cls(model=model)
            
            def ask_model(question_number):
                start_time = time.time()
                try:
                    response = llm.ask(test_prompt.format(question_number=question_number))
                    end_time = time.time()
                    execution_time = end_time - start_time
                    return f"Question {question_number} response: {response}", None, execution_time
                except Exception as e:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    return None, f"Question {question_number} error: {e}", execution_time

            successful_questions = []
            failed_questions = []
            timeout_questions = []
            execution_times = {}

            start_time = time.time()
            
            # 如果 detail 为 True，显示进度条；否则不显示进度条
            if detail:
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
                    future_to_question = {executor.submit(ask_model, i): i for i in range(1, total_questions + 1)}
                    
                    for future in tqdm(concurrent.futures.as_completed(future_to_question), total=total_questions, desc=f"Testing {model}"):
                        question_number = future_to_question[future]
                        try:
                            result, error, exec_time = future.result(timeout=timeout)
                            execution_times[question_number] = exec_time
                            if error:
                                failed_questions.append(question_number)
                            else:
                                successful_questions.append(question_number)
                        except concurrent.futures.TimeoutError:
                            timeout_questions.append(question_number)
                        except Exception as exc:
                            failed_questions.append(question_number)
            else:
                # detail=False 时不显示进度条
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
                    future_to_question = {executor.submit(ask_model, i): i for i in range(1, total_questions + 1)}
                    
                    for future in concurrent.futures.as_completed(future_to_question):
                        question_number = future_to_question[future]
                        try:
                            result, error, exec_time = future.result(timeout=timeout)
                            execution_times[question_number] = exec_time
                            if error:
                                failed_questions.append(question_number)
                            else:
                                successful_questions.append(question_number)
                        except concurrent.futures.TimeoutError:
                            timeout_questions.append(question_number)
                        except Exception as exc:
                            failed_questions.append(question_number)

            end_time = time.time()
            total_time = end_time - start_time

            # 计算模型的 RPS 和成功率
            rps = len(successful_questions) / total_time if total_time > 0 else 0
            success_rate = len(successful_questions) / total_questions

            models_data.append({
                'model': model,
                'rps': rps,
                'success_rate': success_rate,
                'successful_queries': len(successful_questions),
                'failed_queries': len(failed_questions),
                'timeout_queries': len(timeout_questions),
                'total_time': total_time
            })

            if detail:
                # 打印详细的测试报告
                print(f"\n--- Test Report for {model} ---")
                print(f"Successful queries: {len(successful_questions)}")
                print(f"Failed queries: {len(failed_questions)}")
                print(f"Timeout queries: {len(timeout_questions)}")

                if execution_times:
                    average_time = sum(execution_times.values()) / len(execution_times)
                    print(f"\nAverage execution time per query: {average_time:.2f} seconds")
                    print(f"Total execution time: {total_time:.2f} seconds")
                    print(f"Requests per second (RPS): {rps:.2f}")
                else:
                    print("\nNo successful executions to calculate statistics.")

        # 使用评分策略对模型进行排序
        if policy is None:
            policy = cls.default_policy

        sorted_models = policy(models_data)

        if detail:
            # 打印排序后的模型
            print("\n--- Model Ranking ---")
            for i, model in enumerate(sorted_models, 1):
                print(f"Rank {i}: {model['model']} - Score: {model['score']:.4f}, RPS: {model['rps']:.2f}, Success Rate: {model['success_rate']:.2%}")

        best_model = sorted_models[0]['model']

        if apply:
            cls.model = best_model
            print(f"\nBest performing model '{best_model}' has been applied to the current MultiLLM instance.")

        return [model['model'] for model in sorted_models]

    @staticmethod
    def default_policy(models_data, rps_weight=0.6, success_rate_weight=0.4):
        """
        默认评分策略：根据速率和成功率对模型进行评分，并返回按评分排序的模型列表。
        
        速率进行归一化，成功率不归一化。
        
        :param models_data: 模型数据的列表，格式为：
                            [{'model': 'model_name', 'rps': rps_value, 'success_rate': success_rate_value}, ...]
        :param rps_weight: 速率的权重，默认值为 0.6
        :param success_rate_weight: 成功率的权重，默认值为 0.4
        :return: 按评分排序的模型列表，从高到低
        """
        # 提取所有模型的 RPS 和成功率
        rps_values = [model['rps'] for model in models_data]

        # 找到最大 RPS
        max_rps = max(rps_values) if rps_values else 1

        # 计算每个模型的评分
        for model in models_data:
            # 速率归一化
            rps_score = model['rps'] / max_rps if max_rps > 0 else 0
            success_rate_score = model['success_rate']  # 成功率不归一化

            # 综合评分
            model['score'] = (rps_weight * rps_score) + (success_rate_weight * success_rate_score)

        # 按评分从高到低排序
        sorted_models = sorted(models_data, key=lambda x: x['score'], reverse=True)

        return sorted_models

    def show_model(self):
        str_list = [
        'gpt-4o-2024-08-06',
        'chatgpt-4o-latest',
        'deepseek-coder',
        'glm-4-0520',
        'llama-3.1-405b',
        'claude-3-5-sonnet-20240620'
        ]
        for model in str_list:
            print(model+'\n')

    def ask(self, prompt, temperature=0.7):

        url = "https://api.openai-next.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."}, 
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature
        }
        return self._make_request(url, headers, data, 'ask')

    def look(self, image_path, prompt="What's in this image?"):
        url = "https://cn.api.openai-next.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.api_key}"
        }

        base64_image = self._encode_image(image_path)

        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        }

        return self._make_request(url, headers, payload, 'look')

    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def embed_text(self, input_text):
        url = "https://cn.api.openai-next.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)"
        }
        data = {
            "model": self.embed_model,
            "input": input_text
        }
        
        return self._make_request(url, headers, data, 'embed')

    def embed_list(self, texts, num_threads=200):
        def process_text(text):
            embedding = self.embed_text(text)
            return text, embedding

        embeddings = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_text = {executor.submit(process_text, text): text for text in texts}
            for future in tqdm(concurrent.futures.as_completed(future_to_text), total=len(texts), desc="Embedding texts"):
                text = future_to_text[future]
                try:
                    _, embedding = future.result()
                    embeddings[text] = embedding
                except Exception as exc:
                    embeddings[text] = None
                    print(f'Text {text} generated an exception: {exc}')
        
        return embeddings
    
    def partition_by_similarity(self, embeddings_dict, threshold=0.8):
        def cosine_similarity_matrix(matrix):
            norm = np.linalg.norm(matrix, axis=1)
            return np.dot(matrix, matrix.T) / np.outer(norm, norm)

        keys = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[key] for key in keys])

        similarity_matrix = cosine_similarity_matrix(embeddings)
        np.fill_diagonal(similarity_matrix, 0)

        result = {}
        valid_indices = set(range(len(keys)))

        for i in range(len(keys)):
            if i not in valid_indices:
                continue

            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            similar_keys = [keys[j] for j in similar_indices if j in valid_indices]

            for idx in similar_indices:
                valid_indices.discard(idx)

            result[keys[i]] = {'Similar_keys': similar_keys}

        return result

    def calculate_similarity(self, text1, text2):
        embedding1 = self.embed_text(text1)
        embedding2 = self.embed_text(text2)

        if embedding1 is None or embedding2 is None:
            return None

        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity

    def _make_request(self, url, headers, data, request_type):
        try:
            # Encode headers to utf-8
            encoded_headers = {k: v.encode('utf-8') if isinstance(v, str) else v for k, v in headers.items()}
            response = requests.post(url, headers=encoded_headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            
            if request_type == 'ask' or request_type == 'look':
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    return response_json['choices'][0]['message']['content']
                else:
                    raise ValueError("No response content found in the API response.")
            elif request_type == 'embed':
                if 'data' in response_json and len(response_json['data']) > 0:
                    return response_json['data'][0]['embedding']
                else:
                    raise ValueError("No embedding found in the API response.")
            else:
                raise ValueError(f"Unknown request type: {request_type}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error occurred during the API request: {e}")