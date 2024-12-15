import re
import os
import json
from typing import List, Dict, Optional, Any

import PyPDF2


class PDFProcessor:
    """
    A class to process PDF files, extract specific sections based on predefined chapter titles,
    handle translations between Chinese and English section headers, and organize the extracted
    content into a structured JSON format, including the character ranges of each section.
    """

    def __init__(
        self,
        translations: Optional[Dict[str, List[str]]] = None,
        chapters: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initializes the PDFProcessor with translation mappings and chapter definitions.
        If not provided, default translations and chapters are used.

            Args:
                translations (Optional[Dict[str, List[str]]]): A dictionary mapping English section names to
                                                              their possible Chinese equivalents.
                chapters (Optional[Dict[str, List[str]]]): A dictionary mapping standard chapter names to their
                                                         possible aliases in English.
        """
        # Define default translations if none provided
        if translations is None:
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
        else:
            self.translations = translations

        # Define default chapters if none provided
        if chapters is None:
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
        else:
            self.chapters = chapters

        # Precompute normalized chapter titles for efficient matching
        self.normalized_chapters = {
            k: [self.normalize_title(t) for t in v] for k, v in self.chapters.items()
        }

        # Flatten the list of all possible section titles for easy searching
        self.section_titles = [title for titles in self.normalized_chapters.values() for title in titles]

    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalizes a title by removing non-alphanumeric characters and converting to lowercase.

            Args:
                title (str): The title to normalize.

            Returns:
                str: The normalized title.
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().strip()

    @staticmethod
    def extract_section_number(title: str) -> str:
        """
        Attempts to extract the section number from a title.

            Args:
                title (str): The title from which to extract the section number.

            Returns:
                str: The extracted section number or an empty string if not found.
        """
        match = re.match(r'(\d+(?:\.\d+)*)\b', title)
        return match.group(1) if match else ""

    def translate(self, content: str) -> str:
        """
        Detects if the content contains Chinese section headers and translates them to English.

            Args:
                content (str): The text content to translate.

            Returns:
                str: The translated content if Chinese is detected; otherwise, the original content.
        """
        cnt = 0
        for k, vs in self.translations.items():
            for v in vs:
                if v in content:
                    cnt += 1
        if cnt < 3:
            return content

        print("检测到中文章节标题，正在翻译...")

        # Remove content after specific Chinese sections
        for v in ["表格索引", "插图索引"]:
            if v in content[len(content) // 2:]:
                content = content[:content.rindex(v, len(content) // 2)]

        for section in ["acknowledgments", "appendices", "references"]:
            for v in self.translations.get(section, []):
                if v in content[len(content) // 2:]:
                    content = content[:content.rindex(v, len(content) // 2)]
                    break

        # Replace Chinese section headers with English equivalents
        for eng, chinese_titles in self.translations.items():
            for ch_title in chinese_titles:
                pattern = re.compile(r'\b' + re.escape(ch_title) + r'\b', re.IGNORECASE)
                content = pattern.sub(f'\n{eng}\n', content)
        return content

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF file.

            Args:
                pdf_path (str): The path to the PDF file.

            Returns:
                str: The extracted text from the PDF.
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_path)
        except Exception as e:
            print(f"无法读取PDF文件 '{pdf_path}': {e}")
            return ""

        num_pages = len(pdf_reader.pages)
        extracted_text = ""

        for page_number in range(num_pages):
            try:
                page = pdf_reader.pages[page_number]
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "\n"
            except Exception as e:
                print(f"读取页面 {page_number + 1} 时出错: {e}")

        return extracted_text

    def filter_iclr_pdf(self, content: str) -> List[Dict[str, Any]]:
        """
        Filters the PDF content to extract only the specified chapters, along with their ranges.

            Args:
                content (str): The full text content of the PDF.

            Returns:
                List[Dict[str, Any]]: A list of dictionaries, each containing chapter name, text, start, and end.
        """
        content = self.translate(content)
        paper_sections = []

        # Find all matches for chapter titles
        for section, titles in self.normalized_chapters.items():
            for title in titles:
                # Using word boundaries and allowing possible numbering before titles
                pattern = re.compile(r'(?:(?:\d+\.)*\d+\s+)?\b' + re.escape(title) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(content):
                    start_pos = match.start()
                    paper_sections.append({
                        'section': section,
                        'start': start_pos
                    })
                    break  # Only consider the first occurrence of the title

        if not paper_sections:
            print("未找到任何章节标题。")
            return []

        # Sort sections by their start positions
        paper_sections = sorted(paper_sections, key=lambda x: x['start'])

        # Remove duplicates: in case multiple titles map to the same section
        unique_sections = []
        seen = set()
        for sec in paper_sections:
            if sec['section'] not in seen:
                unique_sections.append(sec)
                seen.add(sec['section'])

        # Assign end positions
        for i in range(len(unique_sections)):
            current_section = unique_sections[i]
            start = current_section['start']
            if i < len(unique_sections) - 1:
                end = unique_sections[i + 1]['start']
            else:
                end = len(content)
            extracted_text = content[start:end].strip()
            current_section['end'] = end
            current_section['length'] = len(extracted_text)
            current_section['text'] = extracted_text

        return unique_sections

    def process_pdf(self, pdf_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Processes a single PDF file to extract and filter its chapters, including their ranges.

            Args:
                pdf_path (str): The path to the PDF file.

            Returns:
                Optional[Dict[str, Dict[str, Any]]]: A dictionary mapping chapter names to their extracted text and ranges.
                Returns None if processing fails.
        """
        extracted_text = self.extract_text_from_pdf(pdf_path)
        if not extracted_text:
            return None

        print(f"Processing '{os.path.basename(pdf_path)}'...")

        # Optional: Clean specific formatting issues
        extracted_text = extracted_text.replace("I NTRODUCTION", "INTRODUCTION")
        extracted_text = extracted_text.replace("C ONCLUSION", "CONCLUSION")

        split_sections = self.filter_iclr_pdf(extracted_text)
        if not split_sections:
            return None

        # Organize sections into a dictionary
        section_dict = {}
        for sec in split_sections:
            section_name = sec['section']
            section_dict[section_name] = {
                'text': sec['text'],
                'start': sec['start'],
                'end': sec['end'],
                'length': sec['length']
            }

        return section_dict

    def process_directory_to_json(self, pdf_dir: str, output_json_path: str = 'output.json') -> None:
        """
        Processes all PDF files in a specified directory and writes the extracted information to a JSON file.

            Args:
                pdf_dir (str): The path to the directory containing PDF files.
                output_json_path (str): The path where the JSON output will be saved.
        """
        if not os.path.isdir(pdf_dir):
            print(f"目录 '{pdf_dir}' 不存在。")
            return

        result = {}

        for file_name in os.listdir(pdf_dir):
            if file_name.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, file_name)
                if os.path.exists(file_path):
                    section_data = self.process_pdf(file_path)
                    if section_data:
                        result[file_name] = section_data
                    else:
                        print(f"未能处理文件 '{file_name}'。")
                else:
                    print(f"文件 '{file_path}' 不存在。")

        # Write the result to a JSON file
        try:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(result, json_file, ensure_ascii=False, indent=4)
            print(f"已成功将结果保存到 '{output_json_path}'。")
        except Exception as e:
            print(f"写入JSON文件时出错: {e}")

