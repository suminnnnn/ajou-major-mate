from typing import List, Dict

def format_documents(docs: List[Dict]) -> List[str]:
    formatted = []
    for doc in docs:
        content = doc.get("text", "").strip()
        metadata = doc.get("metadata", {})
        domain = metadata.get("domain", "unknown")
        xml = f"<document><content>{content}</content><department>{domain}</department></document>\n\n"
        formatted.append(xml)
    return formatted