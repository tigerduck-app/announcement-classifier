"""Turn a Moodle forum post into the text the classifier was trained on.

post_text(subject, message) is the title, a blank line and the HTML body as Markdown, the form
export_announcements.py wrote the training data in. model_input() then swaps URLs, emails and
student IDs for the placeholders the training data carries. Names need the course's member list
(export_announcements.course_masker) and stay; on gold that costs nothing measurable.
"""

import re

import html2text
from bs4 import BeautifulSoup

ID_TAG, NAME_TAG, EMAIL_TAG, URL_TAG = "[學號]", "[姓名]", "[EMAIL]", "[網址]"
STUDENT_ID = re.compile(r"(?<![A-Za-z0-9_])\w\d{8}(?!\d)", re.ASCII)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.ASCII)
URL = re.compile(r"(?:https?://|www\.)[\x21-\x7e]+")
MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")


def markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        img.replace_with("[圖片]")
    # shortcut: <pre> here only wraps text pasted from Google Translate; keep it if code blocks show up.
    for pre in soup.find_all("pre"):
        pre.name = "p"
    h = html2text.HTML2Text()
    h.body_width = 0
    text = re.sub(r"[ \t]+\n", "\n", h.handle(str(soup)).replace("\xa0", " "))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def post_text(subject: str, message: str) -> str:
    return f"{subject}\n\n{markdown(message)}".strip()


def _link(m: re.Match) -> str:
    text, target = m[1], m[2]
    if URL.fullmatch(text) or target.startswith("mailto:"):
        return text
    return f"{text} {URL_TAG}".strip()


def mask_contacts(text: str) -> str:
    text = AUTOLINK.sub(r"\1", MD_LINK.sub(_link, text))
    return EMAIL.sub(EMAIL_TAG, URL.sub(URL_TAG, text))


def model_input(text: str) -> str:
    return STUDENT_ID.sub(ID_TAG, mask_contacts(text))


if __name__ == "__main__":
    assert markdown('<p>a<img src="x.png"></p>') == "a[圖片]"
    assert markdown("<p>a<br>b</p><pre> </pre><pre>*c</pre>") == "a\nb\n\n*c"
    assert post_text("HW1", "<p>due <b>12/1</b></p>") == "HW1\n\ndue **12/1**"
    once = model_input("[表單](https://a.tw/x) <https://b.tw> 寄 [a@b.tw](mailto:a@b.tw) 或 www.c.tw，B11132005。")
    assert once == "表單 [網址] [網址] 寄 [EMAIL] 或 [網址]，[學號]。"
    assert model_input(once) == once
    print("moodle_text ok")
