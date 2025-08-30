import os
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

# 定义下载函数，包含重试逻辑
def download_shard(url: str, output_dir: str, max_retries: int = 3):
    """
    下载单个文件分片，并在失败时进行重试。

    Args:
        url (str): 要下载的文件的URL。
        output_dir (str): 保存文件的目录。
        max_retries (int): 最大重试次数。

    Returns:
        tuple: (url, status, message)
               status可以是 'success', 'failed', 'skipped'
    """
    # 从URL中提取文件名
    try:
        filename = url.split('/')[-1].strip()
        if not filename:
            return (url, "failed", "无法从URL中提取有效的文件名")
        
        output_path = os.path.join(output_dir, filename)

        # 如果文件已存在，可以选择跳过
        if os.path.exists(output_path):
            return (url, "skipped", f"文件已存在于 {output_path}")

    except Exception as e:
        return (url, "failed", f"处理URL或路径时出错: {e}")

    # 重试循环
    for attempt in range(max_retries):
        try:
            # 使用 stream=True 进行流式下载，适合大文件
            response = requests.get(url, stream=True, timeout=30)
            # 如果服务器返回错误状态码 (如 404, 500), 抛出异常
            response.raise_for_status()

            # 获取文件总大小，用于tqdm进度条（如果服务器提供了Content-Length）
            total_size = int(response.headers.get('content-length', 0))

            # 以二进制写入模式打开文件
            with open(output_path, 'wb') as f:
                # 使用iter_content逐块写入，避免内存占用过高
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 如果下载成功，返回成功状态并退出循环
            return (url, "success", f"成功下载到 {output_path}")

        except requests.exceptions.RequestException as e:
            # 捕获网络相关的异常
            wait_time = 2 ** attempt  # 指数退避策略，等待时间: 1, 2, 4, ... 秒
            error_message = f"尝试 {attempt + 1}/{max_retries} 失败: {e}. 将在 {wait_time} 秒后重试..."
            # print(f"\n[警告] {url}: {error_message}") # 如果需要在下载过程中看到实时错误，可以取消此行注释
            time.sleep(wait_time)
            
    # 如果所有重试都失败了
    return (url, "failed", f"经过 {max_retries} 次尝试后仍然失败")


def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="从链接文件并行下载数据集分片。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("link_file", type=str, help="包含下载链接的文本文件路径，每行一个链接。")
    parser.add_argument("output_dir", type=str, help="保存下载文件的目录。")
    parser.add_argument(
        "-w", "--workers", type=int, default=os.cpu_count() or 4,
        help="并发下载的线程数 (默认: 系统CPU核心数或4)。"
    )
    parser.add_argument(
        "-r", "--retries", type=int, default=3,
        help="单个文件下载失败后的最大重试次数 (默认: 3)。"
    )

    args = parser.parse_args()

    # 2. 准备工作
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 读取链接文件
    try:
        with open(args.link_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        if not urls:
            print(f"错误: 链接文件 '{args.link_file}' 为空或不包含有效链接。")
            return
    except FileNotFoundError:
        print(f"错误: 链接文件 '{args.link_file}' 未找到。")
        return

    # 按照文件名过滤，按类随机采样
    selected_urls = []
    for url in urls:
        if "quality=high/kind=synthetic" in url and random.random() < 1/60:
            selected_urls.append(url)
    # 增加前缀
    urls = ["https://data.commoncrawl.org/" + url for url in selected_urls]

    # 3. 使用ThreadPoolExecutor执行并行下载
    failed_downloads = []
    skipped_downloads = []
    success_count = 0

    print(f"开始下载... 共 {len(urls)} 个文件, 使用 {args.workers} 个并发线程。")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有下载任务
        future_to_url = {executor.submit(download_shard, url, args.output_dir, args.retries): url for url in urls}

        # 使用tqdm创建进度条，并处理已完成的任务
        progress_bar = tqdm(as_completed(future_to_url), total=len(urls), desc="总体进度", unit="file")
        
        for future in progress_bar:
            try:
                url, status, message = future.result()
                if status == "success":
                    success_count += 1
                elif status == "failed":
                    failed_downloads.append((url, message))
                elif status == "skipped":
                    skipped_downloads.append((url, message))

            except Exception as e:
                # 捕获任务执行本身可能产生的未知异常
                url = future_to_url[future]
                failed_downloads.append((url, f"一个未知的任务执行错误: {e}"))

    # 4. 打印最终总结
    print("\n" + "="*20 + " 下载完成 " + "="*20)
    print(f"成功: {success_count}")
    print(f"跳过 (已存在): {len(skipped_downloads)}")
    print(f"失败: {len(failed_downloads)}")

    if failed_downloads:
        print("\n--- 失败的下载链接 ---")
        for url, reason in failed_downloads:
            print(f"- {url}\n  原因: {reason}")
        
        # (可选) 将失败的链接保存到文件，方便重新下载
        failed_links_file = "failed_links.txt"
        with open(failed_links_file, 'w') as f:
            for url, _ in failed_downloads:
                f.write(url + '\n')
        print(f"\n所有失败的链接已保存到 '{failed_links_file}' 文件中。")


if __name__ == "__main__":
    main()

# nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python parallel_downloader.py data-jsonl.paths /prodcpfs/user/fengmingquan/dataset/raw/nemotron-cc-hq-small -w 32 -r 5 > log/parallel_downloader_1.log 2>&1 &