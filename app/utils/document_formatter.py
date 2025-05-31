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

def format_curriculum_documents(docs: List[Dict]) -> List[str]:
    formatted = []
    for doc in docs:
        content = doc.get("text", "").strip()
        metadata = doc.get("metadata", {})
        domain = metadata.get("domain", "curriculum")
        doc_type = metadata.get("type", "text")

        xml = f"<document><content>{content}</content><department>{domain}</department>"

        if doc_type == "table":
            image_url = metadata.get("image_url", "")
            xml += f"<image_url>{image_url}</image_url>"

        xml += "</document>"
        formatted.append(xml)

    return formatted