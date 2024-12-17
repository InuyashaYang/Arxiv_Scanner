import json
import os
import concurrent.futures
from tenacity import retry, stop_after_attempt, wait_exponential

class Reviewer:
    def __init__(self, llm):
        self.llm = llm
        self.translations = {
            "abstract": ["中文摘要", "摘要", "概要", "要旨"],
            "introduction": ["引言", "简介", "导论"],
            "related work": ["相关工作", "相关研究", "文献回顾", "研究背景"],
            "methodology": ["方法", "研究方法"],
            "experiments": ["实验", "实验研究", "实验设计", "实验部分"],
            "results": ["结果", "研究成果", "实验结果", "结果分析"],
            "discussion": ["讨论", "结果讨论", "讨论部分", "讨论与分析"],
            "conclusion": ["结论", "研究结论", "结论部分", "总结"],
            "future work": ["未来工作", "后续工作", "未来研究方向", "未来研究计划"],
            "acknowledgments": ["致谢", "感谢", "致谢部分", "谢辞"],
            "references": ["参考文献", "参考书目", "文献引用", "引用文献"],
            "appendices": ["附录", "附表", "附加材料", "补充材料", "其他内容"]
        }

        self.chapters = {
            "Abstract": ["abstract"],
            "Introduction": ["introduction", "intro"],
            "Related Work": [
                "related work", "literature review", "background",
                "prior art", "state of the art"
            ],
            "Methodology": [
                "methodology", "methods", "approach",
                "algorithm", "framework", "model"
            ],
            "Experiments": [
                "experiments", "experimental setup", "experimentation",
                "experimental study", "evaluation", "validation"
            ],
            "Results": [
                "results", "findings", "outcomes",
                "data analysis"
            ],
            "Discussion": [
                "discussion", "analysis", "discussion and analysis",
                "interpretation", "insights"
            ],
            "Conclusion": [
                "conclusion", "concluding remarks", "summary",
                "final thoughts", "wrap-up"
            ],
            "Future Work": [
                "future work", "directions for future research",
                "prospects"
            ],
            "Acknowledgments": [
                "acknowledgments", "acknowledgements", "thanks"
            ],
            "References": [
                "references", "bibliography", "works cited"
            ],
            "Appendices": [
                "appendices", "appendix", "supplementary material",
                "additional information"
            ]
        }
        self.max_length_dict = {
            "Abstract": 512,
            "Introduction": 1024,
            "Related Work": 1024,
            "Methodology": 1024,
            "Experiments": 1024,
            "Results": 1024,
            "Discussion": 1024,
            "Conclusion": 512,
            "Future Work": 512,
            "Acknowledgments": 256,
            "References": 2048,
            "Appendices": 1024
        }

    def load_json_data(self, json_file_path):
        """
        加载JSON格式的数据。
        参数:
            json_file_path (str): JSON文件的路径
        返回:
            dict: 解析后的数据
        """
        print(f"开始加载JSON数据: {json_file_path}")
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载JSON数据，共包含 {len(data)} 个文件。")
            return data
        except Exception as e:
            print(f"加载JSON数据时出错: {e}")
            return {}

    def identify_and_extract_sections(self, file_data):
        """
        确认并提取文件中的各个章节内容。
        参数:
            file_data (dict): 单个文件的章节数据
        返回:
            dict: { "标准章节名": "章节文本" }
        """
        print("开始识别并提取章节内容。")
        extracted_sections = {}
        
        # 将file_data的键转换为小写，便于不区分大小写的匹配
        lower_file_data = {k.lower(): v for k, v in file_data.items()}

        for std_chapter, synonyms in self.chapters.items():
            std_chapter_lower = std_chapter.lower()
            for synonym in synonyms:
                synonym_lower = synonym.lower()
                # 直接使用同义词进行匹配
                if synonym_lower in lower_file_data:
                    extracted_sections[std_chapter] = lower_file_data[synonym_lower].get("text", "")
                    print(f"已提取章节: {std_chapter}（使用同义词: {synonym}）")
                    break
                # 查找中文翻译，需要确保翻译列表也转换为小写
                elif synonym_lower in [s.lower() for s in self.translations.get(std_chapter_lower, [])]:
                    extracted_sections[std_chapter] = lower_file_data[synonym_lower].get("text", "")
                    print(f"已提取章节: {std_chapter}（使用翻译: {synonym}）")
                    break
            # 如果没有匹配到任何同义词，跳过该章节
            if std_chapter not in extracted_sections:
                print(f"未找到章节: {std_chapter}")
                continue
        
        print(f"提取完成，共提取到 {len(extracted_sections)} 个章节。")
        return extracted_sections

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def compress_section_text(self, section_name, section_text, llm, max_length_tokens=1024):
        """
        使用大模型将超长章节文本压缩为简短摘要。

        参数:
            section_name (str): 章节名称，如 "Abstract"、"Introduction" 等
            section_text (str): 原始章节文本
            llm: 与大模型交互的实例，提供ask函数调用: llm.ask(prompt)
            max_length_tokens (int): 摘要后希望的最大Token数（或通过字数限制控制）

        返回:
            str: 压缩后的章节文本摘要
        """
        print(f"开始压缩章节: {section_name}")
        prompt_templates = {
            "Abstract": f"请将以下摘要简明扼要地总结，重点突出关键贡献和研究发现：\n\n{section_text}\n\n简要总结：",
            "Introduction": f"请总结以下引言部分，重点突出研究问题、目标和重要性：\n\n{section_text}\n\n简要总结：",
            "Methodology": f"请总结以下方法论部分，详细描述研究方法、技术和实验设计：\n\n{section_text}\n\n简要总结：",
            "Experiments": f"请总结以下实验部分，包括实验设置、使用的数据集、评估指标和主要结果：\n\n{section_text}\n\n简要总结：",
            "Results": f"请总结以下结果部分，强调主要发现、数据分析和性能指标：\n\n{section_text}\n\n简要总结：",
            "Discussion": f"请总结以下讨论部分，分析结果的意义、研究的局限性和未来研究方向：\n\n{section_text}\n\n简要总结：",
            "Conclusion": f"请总结以下结论部分，概述主要结论、研究意义和未来工作：\n\n{section_text}\n\n简要总结：",
            "Future Work": f"请总结以下未来工作部分，描述未来的研究方向和发展计划：\n\n{section_text}\n\n简要总结：",
            "Acknowledgments": f"请总结以下致谢部分，列出感谢的个人或机构：\n\n{section_text}\n\n简要总结：",
            "References": f"请列出以下参考文献的主要来源和贡献：\n\n{section_text}\n\n关键信息：",
            "Appendices": f"请总结以下附录部分的主要内容和补充信息：\n\n{section_text}\n\n简要总结："
        }

        prompt = prompt_templates.get(section_name, f"请将以下内容简明扼要地总结：\n\n{section_text}\n\n简要总结：")
        try:
            summary = llm.ask(prompt)
            print(f"章节压缩完成: {section_name}")
            return summary.strip()
        except Exception as e:
            print(f"压缩章节 '{section_name}' 时出错: {e}")
            return section_text  # 返回原文以防止数据丢失

    def compress_sections(self, sections_dict):
        """
        对章节文本进行长度控制，过长则压缩为摘要。
        参数:
            sections_dict (dict): { "标准章节名": "章节文本" }
        返回:
            dict: { "标准章节名": "处理后的章节文本" }
        """
        print("开始对章节文本进行长度控制。")
        processed_sections = {}

        def process_single_section(section, text):
            print(f"检查章节: {section}")
            tokens = len(text)
            allowed_length = self.max_length_dict.get(section, 1024)
            print(f"章节 '{section}' 长度: {tokens}，允许的最大长度: {allowed_length}")
            
            if tokens > allowed_length:
                print(f"章节 '{section}' 超过最大长度，开始压缩。")
                compressed_text = self.compress_section_text(section, text, self.llm, max_length_tokens=allowed_length)
                print(f"章节 '{section}' 压缩完成。")
                return (section, compressed_text)
            else:
                print(f"章节 '{section}' 长度在允许范围内，无需压缩。")
                return (section, text)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_section = {executor.submit(process_single_section, section, text): section for section, text in sections_dict.items()}
            for future in concurrent.futures.as_completed(future_to_section):
                section, processed_text = future.result()
                processed_sections[section] = processed_text
        
        print("章节长度控制完成。")
        return processed_sections

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_single_note(self, section, text):
        """
        生成单个章节的摘要、优点、缺点和问题。

        参数:
            section (str): 章节名称
            text (str): 章节文本

        返回:
            tuple: (summary, strength, weakness, question)
        """
        local_summaries = {}
        local_strengths = []
        local_weaknesses = []
        local_questions = []
        print(f"处理章节: {section}")
        try:
            # 按章节生成摘要
            prompt_summary = f"请将以下{section}部分简明扼要地总结，突出关键内容：\n\n{text}\n\n简要总结："
            summary = self.llm.ask(prompt_summary)
            local_summaries[section] = summary
            print(f"摘要生成完成: {section}")
        except Exception as e:
            print(f"生成摘要时出错: {section} - {e}")
            local_summaries[section] = ""
        
        try:
            # 按章节生成优点
            prompt_strength = f"请根据以下{section}部分内容，列出该部分的优点：\n\n{text}\n\n优点："
            strength = self.llm.ask(prompt_strength)
            local_strengths.append(strength)
            print(f"优点生成完成: {section}")
        except Exception as e:
            print(f"生成优点时出错: {section} - {e}")
            local_strengths.append("")
        
        try:
            # 按章节生成缺点
            prompt_weakness = f"请根据以下{section}部分内容，列出该部分的缺点：\n\n{text}\n\n缺点："
            weakness = self.llm.ask(prompt_weakness)
            local_weaknesses.append(weakness)
            print(f"缺点生成完成: {section}")
        except Exception as e:
            print(f"生成缺点时出错: {section} - {e}")
            local_weaknesses.append("")
        
        try:
            # 按章节生成问题
            prompt_question = f"请根据以下{section}部分内容，提出相关的问题：\n\n{text}\n\n问题："
            question = self.llm.ask(prompt_question)
            local_questions.append(question)
            print(f"问题生成完成: {section}")
        except Exception as e:
            print(f"生成问题时出错: {section} - {e}")
            local_questions.append("")

        return (local_summaries, local_strengths, local_weaknesses, local_questions)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate_single_score(self, metric, review_text):
        """
        生成单一评分。

        参数:
            metric (str): 评分维度
            review_text (str): 综合评审文本

        返回:
            tuple: (metric, score)
        """
        print(f"生成评分: {metric}")
        prompt_score = (
            f"请根据论文内容和以下综合评审，给出以下维度的评分，并简要说明理由。\n\n"
            f"综合评审：\n{review_text}\n\n"
            f"维度：{metric}\n\n"
            f"评分（取1到5之间的整数）\n\n"
            f"请以如下格式返回:\n分数: score\n原因:..."
        )
        try:
            score = self.llm.ask(prompt_score)
            print(f"评分完成: {metric} - {score}")
            return (metric, score)
        except Exception as e:
            print(f"生成评分时出错: {metric} - {e}")
            return (metric, "")

    def generate_notes(self, processed_sections):
        """
        生成阅读笔记，包括摘要、优点、缺点和问题。
        参数:
            processed_sections (dict): { "标准章节名": "处理后的章节文本" }
        返回:
            dict: {
                "Summaries": { "章节名": "摘要" },
                "Strengths": "优点...",
                "Weaknesses": "缺点...",
                "Questions": "问题..."
            }
        """
        print("开始生成阅读笔记。")
        summaries = {}
        strengths = []
        weaknesses = []
        questions = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate_single_note, section, text): section for section, text in processed_sections.items()}
            for future in concurrent.futures.as_completed(futures):
                local_summaries, local_strengths, local_weaknesses, local_questions = future.result()
                summaries.update(local_summaries)
                strengths.extend(local_strengths)
                weaknesses.extend(local_weaknesses)
                questions.extend(local_questions)
        
        # 综合各章节的优点、缺点和问题
        combined_strengths = " ".join(strengths)
        combined_weaknesses = " ".join(weaknesses)
        combined_questions = " ".join(questions)
        
        print("阅读笔记生成完成。")
        return {
            "Summaries": summaries,
            "Strengths": combined_strengths,
            "Weaknesses": combined_weaknesses,
            "Questions": combined_questions
        }

    def generate_review_report(self, notes):
        """
        根据阅读笔记生成最终的评审报告。
        参数:
            notes (dict): 生成的阅读笔记
        返回:
            dict: 评审报告，包括各类别和评分
        """
        print("开始生成评审报告。")
        review_report = {}
        print(notes)
        try:
            # 生成综合评审
            prompt_review = (
                f"基于以下阅读笔记，撰写一份全面的论文评审报告。\n\n"
                f"摘要总结：\n{json.dumps(notes['Summaries'], ensure_ascii=False, indent=2)}\n\n"
                f"优点：\n{notes['Strengths']}\n\n"
                f"缺点：\n{notes['Weaknesses']}\n\n"
                f"问题：\n{notes['Questions']}\n\n"
                f"请将以上内容整合成一份连贯的评审报告。"
            )
            review = self.llm.ask(prompt_review)
            review_report['Review'] = review
            print("综合评审生成完成。")
        except Exception as e:
            print(f"生成综合评审时出错: {e}")
            review_report['Review'] = ""

        # 生成评分
        metrics = ["soundness", "presentation", "contribution", "rating", "confidence"]
        scores = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate_single_score, metric, review_report.get('Review', '')): metric for metric in metrics}
            for future in concurrent.futures.as_completed(futures):
                metric, score = future.result()
                scores[metric] = score

        review_report['Scores'] = scores
        print("评审报告生成完成。")
        return review_report

    def process_file(self, file_data):
        """
        处理单个文件的数据，生成评审报告。
        参数:
            file_data (dict): 单个文件的章节数据
            max_length_dict (dict): { "标准章节名": 最大Token数 }
        返回:
            dict: 评审报告
        """
        print("========================================")
        print("开始处理新文件。")
        # Step 1: 确认并提取章节内容
        print("步骤1: 确认并提取章节内容。")
        sections = self.identify_and_extract_sections(file_data)
        
        # Step 2: 对章节文本进行长度控制
        print("步骤2: 对章节文本进行长度控制。")
        processed_sections = self.compress_sections(sections)
        
        # Step 3: 生成阅读笔记
        print("步骤3: 生成阅读笔记。")
        notes = self.generate_notes(processed_sections)
        print(notes)
        
        # Step 4: 生成评审报告
        print("步骤4: 生成评审报告。")
        review_report = self.generate_review_report(notes)
        
        print("文件处理完成。")
        print("========================================\n")
        return review_report