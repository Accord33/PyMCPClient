import asyncio
from typing import Optional, Dict, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # .env ファイルから環境変数を読み込む

class MCPClient:
    """
    MCPサーバーと通信し、Claude APIを使用してクエリを処理するクライアントクラス。
    このクラスは、複数のサーバーとの接続確立、ツールの呼び出し、対話セッションの管理を担当します。
    """
    def __init__(self):
        # クラスの初期化処理
        # 各種属性の初期化と設定
        self.sessions: Dict[str, ClientSession] = {}  # 複数のMCPサーバーセッションを管理
        self.exit_stack = AsyncExitStack()  # 非同期リソースの管理用スタック
        self.anthropic = Anthropic()  # Anthropic APIクライアントの初期化
        self.server_tools: Dict[str, List] = {}  # サーバーごとのツール情報を保存

    async def connect_to_server(self, server_script_path: str):
        """
        MCPサーバーに接続するメソッド
        
        Args:
            server_script_path: サーバースクリプトのパス（.pyまたは.jsファイル）
        
        処理内容:
        1. スクリプトの種類（PythonまたはJavaScript）を判定
        2. 適切なコマンドでサーバープロセスを起動
        3. 標準入出力を通じてサーバーと通信
        4. セッションの初期化と利用可能ツールの取得
        
        Returns:
            str: サーバー識別子
        """
        # スクリプトの拡張子をチェックして種類を判定
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        # スクリプトの種類に応じたコマンドを設定
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # サーバー識別子を生成
        server_id = f"server_{len(self.sessions) + 1}"
        
        # 標準入出力を使用してサーバープロセスと通信するための設定
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        # セッションの初期化
        await session.initialize()
        
        # セッションをディクショナリに保存
        self.sessions[server_id] = session
        
        # 利用可能なツールの一覧を取得して保存
        response = await session.list_tools()
        tools = response.tools
        self.server_tools[server_id] = tools
        
        # 接続したサーバーのツール情報を表示
        print(f"\nConnected to {server_id} with tools:", [tool.name for tool in tools])
        
        return server_id

    async def process_query(self, query: str) -> str:
        """
        Claudeを使用してクエリを処理し、必要に応じてツールを呼び出すメソッド
        
        Args:
            query: ユーザーからの質問やリクエスト
            
        Returns:
            処理結果のテキスト
            
        処理内容:
        1. 全サーバーから利用可能なツールを収集
        2. Claudeに初期クエリを送信
        3. Claudeのレスポンスを解析
        4. ツール呼び出しがある場合は適切なサーバーで実行
        5. ツールの結果をClaudeに返して会話を継続
        6. 最終的な応答を生成
        """
        # 接続サーバーがない場合はエラー
        if not self.sessions:
            return "エラー: サーバーに接続されていません。まずサーバーに接続してください。"
            
        # 会話履歴の初期化
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        # 全サーバーから利用可能なツールの情報を収集
        available_tools = []
        tool_to_server_map = {}  # ツール名からサーバーIDへのマッピング
        
        for server_id, tools in self.server_tools.items():
            for tool in tools:
                # サーバー識別子付きのツール名を生成
                qualified_tool_name = f"{tool.name}"
                # ツール情報を追加
                available_tools.append({
                    "name": qualified_tool_name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
                # ツール名からサーバーIDへのマッピングを保存
                tool_to_server_map[qualified_tool_name] = server_id

        # Claude APIに初回のリクエストを送信
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools  # 全サーバーの利用可能なツールを指定
        )

        # レスポンスの処理とツール呼び出しの処理
        tool_results = []  # ツール呼び出しの結果を保存するリスト
        final_text = []  # 最終的な応答テキストを構築するリスト

        # レスポンスの内容を処理
        for content in response.content:
            if content.type == 'text':
                # テキスト応答の場合はそのまま追加
                final_text.append(content.text)
            elif content.type == 'tool_use':
                # ツール呼び出しの場合は実行して結果を処理
                tool_name = content.name
                tool_args = content.input
                
                # ツールを持つサーバーを特定
                server_id = tool_to_server_map.get(tool_name)
                if not server_id:
                    error_msg = f"エラー: ツール '{tool_name}' に対応するサーバーが見つかりません。"
                    final_text.append(error_msg)
                    continue
                
                # 該当サーバーのセッションを取得
                session = self.sessions[server_id]
                
                # ツールを実行
                try:
                    result = await session.call_tool(tool_name, tool_args)
                    tool_results.append({"call": tool_name, "server": server_id, "result": result})
                    final_text.append(f"[{server_id}のツール {tool_name} を呼び出し中、引数: {tool_args}]")
                    
                    # ツール実行結果を含めて会話を継続
                    if hasattr(content, 'text') and content.text:
                        messages.append({
                            "role": "assistant",
                            "content": content.text
                        })
                    messages.append({
                        "role": "user", 
                        "content": result.content
                    })
                    
                    # ツール実行結果を含めて再度Claudeに問い合わせ
                    response = self.anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=messages,
                    )
                    
                    # 新しい応答を最終テキストに追加
                    final_text.append(response.content[0].text)
                except Exception as e:
                    error_msg = f"ツール実行エラー ({server_id}): {str(e)}"
                    final_text.append(error_msg)

        # 全ての応答を結合して返す
        return "\n".join(final_text)

    async def chat_loop(self):
        """
        対話式のチャットループを実行するメソッド
        
        ユーザーからの入力を受け取り、process_queryメソッドを使用して処理し、
        結果を表示します。'quit'と入力されるまで繰り返します。
        """
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """
        リソースをクリーンアップするメソッド
        
        すべての非同期リソースを適切に解放します。
        """
        await self.exit_stack.aclose()

async def main():
    """
    メインエントリーポイント
    
    複数のサーバースクリプトパスを受け取り、それぞれのサーバーに接続して
    対話セッションを開始します。
    """
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script1> [<path_to_server_script2> ...]")
        sys.exit(1)
        
    client = MCPClient()
    try:
        # コマンドライン引数からすべてのサーバースクリプトパスを取得して接続
        for server_path in sys.argv[1:]:
            await client.connect_to_server(server_path)
            
        # 対話ループを開始
        await client.chat_loop()
    finally:
        # リソースのクリーンアップ
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())