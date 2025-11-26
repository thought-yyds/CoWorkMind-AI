# test_qq_mail.py
import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(__file__))

from QMailMCPTool import list_folders, list_recent_mail

async def test_tools():
    print("=== 测试QQ邮箱MCP工具 ===")
    
    # 测试1: 列出文件夹
    print("\n1. 测试列出文件夹...")
    result = list_folders()
    print(f"文件夹列表: {result}")
    
    # 测试2: 列出最近邮件
    print("\n2. 测试列出最近邮件...")
    result = list_recent_mail(limit=3)
    print(f"最近邮件: {result}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_tools())