from typing import List, Optional
import os
import glob
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()
NAME=os.environ.get("NAME")

# Initialize FastMCP server
mcp = FastMCP("file_finder")

def format_file_list(files: List[str], base_dir: str) -> str:
    """フォーマットされたファイルリストを返す"""
    if not files:
        return "一致するファイルはありません。"
    
    result = f"ディレクトリ '{base_dir}' 内で {len(files)} 件のファイルが見つかりました:\n\n"
    for i, file in enumerate(files, 1):
        # フルパスではなく相対パスを表示する
        relative_path = os.path.relpath(file, base_dir)
        file_size = os.path.getsize(file)
        file_time = os.path.getmtime(file)
        
        # ファイルサイズの単位変換
        size_str = f"{file_size} B"
        if file_size >= 1024:
            size_str = f"{file_size/1024:.2f} KB"
        if file_size >= 1024*1024:
            size_str = f"{file_size/(1024*1024):.2f} MB"
            
        result += f"{i}. {relative_path}\n   サイズ: {size_str}, 最終更新: {int(file_time)}\n"
    
    return result

@mcp.tool()
async def find_files(directory: str = f"/Users/{NAME}/Downloads", pattern: Optional[str] = "*") -> str:
    """特定のディレクトリ内でファイルを検索します。

    Args:
        directory: 検索するディレクトリのパス（デフォルト: /Users/{NAME}/Downloads）
        pattern: 検索パターン（例: "*.txt", "doc*.*"）、指定しない場合は全ファイル
    """
    directory = os.path.expanduser(directory)  # ~を展開する
    
    if not os.path.exists(directory):
        return f"エラー: ディレクトリ '{directory}' が存在しません。"
    
    if not os.path.isdir(directory):
        return f"エラー: '{directory}' はディレクトリではありません。"
    
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    
    # ディレクトリを除外して、ファイルのみをリストに含める
    files = [f for f in files if os.path.isfile(f)]
    
    return format_file_list(files, directory)

@mcp.tool()
async def search_file_content(directory: str = f"/Users/{NAME}/Downloads", search_text: str = "", file_extension: Optional[str] = None) -> str:
    """ディレクトリ内のファイル内容を検索します。

    Args:
        directory: 検索するディレクトリのパス（デフォルト: /Users/{NAME}/Downloads）
        search_text: ファイル内で検索するテキスト
        file_extension: 検索対象のファイル拡張子（例: "txt", "py"）、指定しない場合は全ファイル
    """
    directory = os.path.expanduser(directory)  # ~を展開する
    
    if not os.path.exists(directory):
        return f"エラー: ディレクトリ '{directory}' が存在しません。"
    
    if not os.path.isdir(directory):
        return f"エラー: '{directory}' はディレクトリではありません。"
    
    pattern = "*"
    if file_extension:
        pattern = f"*.{file_extension}"
    
    matching_files = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if file_extension is None or filename.endswith(f".{file_extension}"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        content = f.read()
                        if search_text in content:
                            matching_files.append(filepath)
                except Exception:
                    # バイナリファイルなど読み取れないファイルはスキップ
                    pass
    
    if not matching_files:
        return f"'{search_text}' を含むファイルは見つかりませんでした。"
    
    result = f"'{search_text}' を含む {len(matching_files)} 件のファイルが見つかりました:\n\n"
    for i, file in enumerate(matching_files, 1):
        relative_path = os.path.relpath(file, directory)
        result += f"{i}. {relative_path}\n"
    
    return result

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
