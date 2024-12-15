import os
import re
import arxiv
import requests
import logging
from tqdm.auto import tqdm  # 使用 tqdm.auto 以自动选择最佳进度条实现

class ArxivDownloader:
    def __init__(self, download_dir: str, max_results: int = 5, sort_by: str = 'submittedDate', log_file: str = None):
        """
        初始化 ArxivDownloader 类。

        Args:
            download_dir (str): 下载文件保存目录。
            max_results (int): 最大搜索结果数，默认5。
            sort_by (str): 排序方式，默认按提交日期排序。
            log_file (str): 日志文件路径。如果为 None，将使用控制台输出。
        """
        self.download_dir = download_dir
        self.max_results = max_results
        self.sort_by = sort_by

        # 设置日志配置
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # 确保下载目录存在
        os.makedirs(self.download_dir, exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """
        清理文件名，移除非法字符。

        Args:
            filename (str): 原始文件名。

        Returns:
            str: 清理后的文件名。
        """
        sanitized = re.sub(r'[\\/*?:"<>|]', "", filename)
        sanitized = sanitized.strip()
        return sanitized

    def download_pdf(self, pdf_url: str, save_path: str):
        """
        下载PDF文件并保存到指定路径。

        Args:
            pdf_url (str): PDF文件的URL。
            save_path (str): 本地保存路径。
        """
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()  # 确保请求成功

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(save_path))

            with open(save_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                self.logger.warning(f"下载可能未完成: {save_path}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"下载失败: {pdf_url}\n错误信息: {e}")

    def search_and_download(self, keyword: str):
        """
        根据关键词搜索arXiv论文并下载PDF文件。

        Args:
            keyword (str): 搜索关键词。
        """
        # 设置排序标准
        sort_criteria = {
            'relevance': arxiv.SortCriterion.Relevance,
            'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
            'submittedDate': arxiv.SortCriterion.SubmittedDate,
        }

        sort = sort_criteria.get(self.sort_by, arxiv.SortCriterion.SubmittedDate)

        # 使用 arxiv.Search 进行搜索
        self.logger.info(f"正在搜索关键词：'{keyword}'，最多 {self.max_results} 个结果...")
        
        # 创建 arxiv 搜索对象
        search = arxiv.Search(
            query=keyword,
            max_results=self.max_results,
            sort_by=sort,
            sort_order=arxiv.SortOrder.Descending
        )

        # 获取搜索结果
        results = list(search.results())  # 使用 Search 对象来获取并转换为列表
        self.logger.info(f"找到 {len(results)} 篇论文。\n")

        for idx, result in enumerate(results, start=1):
            title = self.sanitize_filename(result.title)
            arxiv_id = result.entry_id.split("/")[-1]  # 获取 arXiv ID
            pdf_url = result.pdf_url
            file_name = f"{arxiv_id} - {title}.pdf"
            save_path = os.path.join(self.download_dir, file_name)

            # 检查文件是否已存在
            if os.path.exists(save_path):
                self.logger.info(f"{idx}/{len(results)} 已存在，跳过下载: {file_name}")
                continue

            self.logger.info(f"{idx}/{len(results)} 正在下载: {file_name}")
            self.download_pdf(pdf_url, save_path)
            self.logger.info(f"{idx}/{len(results)} 下载完成: {file_name}\n")

        self.logger.info("所有下载任务完成。")
