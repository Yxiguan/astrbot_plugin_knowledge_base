<h1 align="center">AstrBot 知识库插件</h1>

<p align="center">
  <strong>为你的 AstrBot 注入专属知识，打造更智能的对话体验！</strong>
</p>


<p align="center">
  <code>astrbot_plugin_knowledge_base</code> 是一个为 <a href="https://github.com/Soulter/AstrBot">AstrBot</a> 聊天机器人框架量身定制的强大知识库插件。
  它允许您的机器人连接到自定义的知识源，通过先进的检索增强生成 (RAG) 技术，
  让机器人能够基于准确、具体的私有信息进行对话，而不仅仅依赖预训练模型的通用知识。
</p>

---

本插件处于公开测试期，期间可能会经历版本的快速迭代。

可以添加QQ群获得更快的帮助:953245617

---
## 🚀 核心功能一览

<div align="center">

| 功能点                                | 描述                                                                 | 图标 |
| :------------------------------------ | :------------------------------------------------------------------- | :--: |
| 🧠 **智能 RAG 集成**                    | 自动检索知识增强 LLM 回答，更精准、更相关                               | 🎯   |
| 💾 **多向量数据库**                     | 支持 Faiss, Milvus Lite, Milvus Server，满足不同规模需求                | 🗂️   |
| ✍️ **灵活内容导入**                     | 通过指令添加文本、本地文件 (`.txt`, `.md`) 或 URL 轻松构建知识库         | 📥   |
| 🗣️ **便捷指令交互**                     | `/kb` 系列指令，覆盖创建、搜索、切换、管理等全流程                       | 💬   |
| ⚙️ **高度可配置**                       | 自定义 Embedding、文本处理、RAG 策略，深度定制您的知识引擎             | 🛠️   |
| 📦 **持久化存储**                       | 知识数据与用户偏好安全存储，插件更新无忧                               | 🛡️   |
| 👥 **会话级知识库**                     | 为不同用户或群组设定专属的默认知识库，实现个性化知识服务               | 🌐   |
| 📄 **文件下载与解析**                   | 直接从 URL 下载并解析文本文件，快速扩充知识                             | 🔗   |

</div>

---

## 🤔 为何选择本插件？

*   🌟 **赋予机器人“记忆”与“专长”**：让您的机器人掌握特定领域知识，如产品手册、项目文档、常见问答库等。
*   🎯 **提升回答质量**：基于事实依据进行回答，有效减少大型语言模型的“幻觉”和不准确性。
*   💡 **个性化互动体验**：根据对话场景或用户群体，动态调用相应知识库，提供更贴心的服务。
*   🔐 **数据自主可控**：知识数据存储在您选择的环境（本地或私有服务器），保障数据安全与隐私。
*   🧩 **与 AstrBot 无缝集成**：专为 AstrBot 设计，安装配置简单，开箱即用。

---

## 🛠️ 安装与配置指南

### 1. 安装插件

   将 `astrbot_plugin_knowledge_base` 文件夹放置到 AstrBot 的 `plugins` 目录下。

### 2. 安装依赖

   根据您选择的向量数据库类型，通过 pip 安装相应依赖：
   *   **Faiss**: `pip install faiss-cpu` (或 `faiss-gpu` 搭配 CUDA)
   *   **Milvus Lite**: `pip install pymilvus`
   *   **Milvus Server**: 确保您有一个正在运行的 Milvus 服务实例。

   💡 **提示**: 如果希望使用 `astrbot_plugin_embedding_adapter` 自动配置 Embedding，请确保该插件也已安装。

---

## 🎮 快速开始：指令体验

通过向机器人发送以 `/kb` (或别名 `知识库`) 开头的指令来与知识库互动：

*   `✨ /kb help` - 显示所有可用指令，您的入门向导。
*   `➕ /kb add text "AstrBot真棒！" my_astr_facts` - 添加文本到 `my_astr_facts` 知识库。
*   `📄 /kb add file https://example.com/faq.txt common_issues` - 从URL添加文件内容。
*   `🔍 /kb search "如何安装插件？" my_astr_facts` - 在知识库中搜索答案。
*   `🚀 /kb use project_alpha` - 将当前会话的默认知识库切换到 `project_alpha`。（在会话中一定要使用这个，否则将不会在提示词中嵌入知识库中的内容）
*   `📋 /kb list` - 查看所有已创建的知识库。
*   `🗑️ /kb delete old_stuff` - (管理员) 删除知识库 `old_stuff` (会有二次确认)。

---

## 💡 RAG 工作流程示意

当您向启用了知识库增强的 AstrBot 提问时：

```
   用户提问 🗣️
       │
       └───┐
           ▼
     🤖 AstrBot 接收请求
           │
   (知识库插件介入)
           │
   1. 🔍 在当前知识库中搜索与提问最相关的信息
           │
   2. 📚 获取相关知识片段
           │
   3. ✍️ 将知识片段 + 用户原始提问 整合进 Prompt
           │
           ▼
     🧠 发送给大语言模型 (LLM) 进行处理
           │
           ▼
   ✨ 生成包含知识库信息的、更优质的回答
```


## 🤝 贡献与支持

欢迎通过 Pull Requests 或 Issues 为本项目贡献代码、提出建议或报告问题！

⭐️ 如果这个插件对您有帮助，请给个 Star 吧！

---

<p align="center">
  由 <a href="https://github.com/lxfight">lxfight</a> 开发与维护
</p>
<p align="center">
  <a href="https://github.com/lxfight/astrbot_plugin_knowledge_base">回到项目仓库</a>
</p>
