"""Editable Streamlit portfolio site with a lightweight JSON admin CMS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


CONTENT_PATH = Path("portfolio_content.json")


def _default_content() -> dict[str, Any]:
    """Return a minimal fallback content payload."""
    return {
        "site": {
            "name": "AI Engineering Portfolio",
            "tagline": "Graph ML, GraphRAG, and applied AI engineering.",
            "one_liner": "GNN, GraphRAG, explainable AI, and production-minded AI engineering projects.",
            "representative_technologies": ["Python", "PyTorch", "GNN", "GraphRAG", "Streamlit"],
            "representative_project_ids": [],
            "contacts": {"email": "", "github": "", "linkedin": "", "notion": "", "resume_pdf": ""},
        },
        "projects": [],
        "research": {
            "title": "Master's Research Summary",
            "summary": "",
            "problem_definition": "",
            "proposed_method": "",
            "experimental_results": "",
            "limitations_and_extensions": "",
        },
        "blog": [],
    }


def load_content() -> dict[str, Any]:
    """Load portfolio content from JSON."""
    if not CONTENT_PATH.exists():
        return _default_content()
    try:
        data = json.loads(CONTENT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_content()
    return data if isinstance(data, dict) else _default_content()


def save_content(content: dict[str, Any]) -> None:
    """Save portfolio content atomically."""
    temp_path = CONTENT_PATH.with_suffix(".tmp.json")
    temp_path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(CONTENT_PATH)


def admin_password() -> str:
    """Read admin password from Streamlit secrets or environment."""
    try:
        secret_value = st.secrets.get("portfolio_admin_password")
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    return os.environ.get("PORTFOLIO_ADMIN_PASSWORD", "admin")


def split_lines(value: str) -> list[str]:
    """Convert newline text to a clean string list."""
    return [line.strip() for line in value.splitlines() if line.strip()]


def join_lines(values: list[Any]) -> str:
    """Convert a list to newline text."""
    return "\n".join(str(item) for item in values or [])


def normalize_project_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert editable project rows back to structured project dictionaries."""
    projects: list[dict[str, Any]] = []
    for _, row in frame.fillna("").iterrows():
        project_id = str(row.get("id", "")).strip()
        title = str(row.get("title", "")).strip()
        if not project_id and not title:
            continue
        projects.append(
            {
                "id": project_id or title.lower().replace(" ", "-"),
                "title": title,
                "subtitle": str(row.get("subtitle", "")).strip(),
                "description": str(row.get("description", "")).strip(),
                "tech": [item.strip() for item in str(row.get("tech", "")).split(",") if item.strip()],
                "links": {
                    "github": str(row.get("github", "")).strip(),
                    "demo": str(row.get("demo", "")).strip(),
                    "paper": str(row.get("paper", "")).strip(),
                },
                "highlights": split_lines(str(row.get("highlights", ""))),
            }
        )
    return projects


def projects_to_frame(projects: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten projects for Streamlit data_editor."""
    rows = []
    for project in projects:
        links = project.get("links", {}) if isinstance(project.get("links"), dict) else {}
        rows.append(
            {
                "id": project.get("id", ""),
                "title": project.get("title", ""),
                "subtitle": project.get("subtitle", ""),
                "description": project.get("description", ""),
                "tech": ", ".join(project.get("tech", []) or []),
                "github": links.get("github", ""),
                "demo": links.get("demo", ""),
                "paper": links.get("paper", ""),
                "highlights": join_lines(project.get("highlights", []) or []),
            }
        )
    return pd.DataFrame(rows)


def normalize_blog_rows(frame: pd.DataFrame) -> list[dict[str, str]]:
    """Convert editable blog rows back to dictionaries."""
    posts: list[dict[str, str]] = []
    for _, row in frame.fillna("").iterrows():
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        posts.append(
            {
                "title": title,
                "category": str(row.get("category", "")).strip(),
                "date": str(row.get("date", "")).strip(),
                "summary": str(row.get("summary", "")).strip(),
                "content": str(row.get("content", "")).strip(),
            }
        )
    return posts


def page_home(content: dict[str, Any]) -> None:
    """Render home page."""
    site = content.get("site", {})
    projects = content.get("projects", [])
    st.title(site.get("name", "Portfolio"))
    st.subheader(site.get("tagline", ""))
    st.write(site.get("one_liner", ""))

    st.markdown("### 대표 기술")
    tech = site.get("representative_technologies", [])
    st.write(" · ".join(tech) if tech else "-")

    st.markdown("### 대표 프로젝트")
    representative_ids = set(site.get("representative_project_ids", []) or [])
    representative = [project for project in projects if project.get("id") in representative_ids]
    if not representative:
        representative = projects[:3]
    columns = st.columns(3)
    for idx, project in enumerate(representative[:3]):
        with columns[idx % 3]:
            st.markdown(f"#### {project.get('title', '')}")
            st.caption(project.get("subtitle", ""))
            st.write(project.get("description", ""))

    st.markdown("### 연락 링크")
    contacts = site.get("contacts", {})
    for label, key in [("Email", "email"), ("GitHub", "github"), ("LinkedIn", "linkedin"), ("Notion", "notion")]:
        value = contacts.get(key)
        if value:
            st.markdown(f"- **{label}**: {value}")


def page_projects(content: dict[str, Any]) -> None:
    """Render projects page."""
    st.title("Projects")
    for project in content.get("projects", []):
        with st.container(border=True):
            st.subheader(project.get("title", "Untitled project"))
            st.caption(project.get("subtitle", ""))
            st.write(project.get("description", ""))
            tech = project.get("tech", [])
            if tech:
                st.markdown("**Tech:** " + ", ".join(tech))
            highlights = project.get("highlights", [])
            if highlights:
                st.markdown("**Highlights**")
                for item in highlights:
                    st.markdown(f"- {item}")
            links = project.get("links", {}) if isinstance(project.get("links"), dict) else {}
            link_text = [f"[{name}]({url})" for name, url in links.items() if url]
            if link_text:
                st.markdown(" · ".join(link_text))


def page_research(content: dict[str, Any]) -> None:
    """Render research page."""
    research = content.get("research", {})
    st.title("Research")
    st.subheader(research.get("title", "Master's Research Summary"))
    sections = [
        ("석사 연구 요약", "summary"),
        ("문제 정의", "problem_definition"),
        ("제안 방법", "proposed_method"),
        ("실험 결과", "experimental_results"),
        ("한계 및 확장 방향", "limitations_and_extensions"),
    ]
    for heading, key in sections:
        st.markdown(f"### {heading}")
        st.write(research.get(key, ""))


def page_blog(content: dict[str, Any]) -> None:
    """Render blog page."""
    st.title("Blog")
    for post in content.get("blog", []):
        with st.expander(f"{post.get('title', 'Untitled')} · {post.get('category', '')} · {post.get('date', '')}", expanded=False):
            st.write(post.get("summary", ""))
            st.markdown(post.get("content", ""))


def page_resume(content: dict[str, Any]) -> None:
    """Render resume page."""
    contacts = content.get("site", {}).get("contacts", {})
    st.title("Resume")
    resume_pdf = contacts.get("resume_pdf")
    if resume_pdf:
        st.markdown(f"[PDF 다운로드]({resume_pdf})")
    else:
        st.info("관리자 페이지에서 resume_pdf 링크를 설정해 주세요.")
    for label, key in [("Email", "email"), ("GitHub", "github"), ("LinkedIn", "linkedin"), ("Notion", "notion")]:
        value = contacts.get(key)
        if value:
            st.markdown(f"- **{label}**: {value}")


def admin_login() -> bool:
    """Authenticate admin session."""
    if st.session_state.get("portfolio_admin_authenticated"):
        return True
    password = st.text_input("Admin password", type="password")
    if st.button("Login"):
        if password == admin_password():
            st.session_state["portfolio_admin_authenticated"] = True
            st.rerun()
        st.error("비밀번호가 올바르지 않습니다.")
    if admin_password() == "admin":
        st.warning("기본 관리자 비밀번호는 'admin'입니다. 운영 환경에서는 PORTFOLIO_ADMIN_PASSWORD 또는 Streamlit secrets를 설정해 주세요.")
    return False


def page_admin(content: dict[str, Any]) -> None:
    """Render admin editing page."""
    st.title("Admin")
    if not admin_login():
        return

    st.success("관리자 모드입니다. 이 페이지에서 모든 콘텐츠를 수정할 수 있습니다.")
    editable = json.loads(json.dumps(content))
    site = editable.setdefault("site", {})
    contacts = site.setdefault("contacts", {})

    tab_site, tab_projects, tab_research, tab_blog, tab_json = st.tabs(["Home/Resume", "Projects", "Research", "Blog", "Raw JSON"])

    with tab_site:
        site["name"] = st.text_input("Site name", value=site.get("name", ""))
        site["tagline"] = st.text_input("Tagline", value=site.get("tagline", ""))
        site["one_liner"] = st.text_area("한 줄 소개", value=site.get("one_liner", ""), height=90)
        site["representative_technologies"] = split_lines(
            st.text_area("대표 기술, 한 줄에 하나씩", value=join_lines(site.get("representative_technologies", [])), height=180)
        )
        project_ids = [project.get("id", "") for project in editable.get("projects", []) if project.get("id")]
        site["representative_project_ids"] = st.multiselect(
            "대표 프로젝트 3개",
            project_ids,
            default=[item for item in site.get("representative_project_ids", []) if item in project_ids],
            max_selections=3,
        )
        contacts["email"] = st.text_input("Email", value=contacts.get("email", ""))
        contacts["github"] = st.text_input("GitHub", value=contacts.get("github", ""))
        contacts["linkedin"] = st.text_input("LinkedIn", value=contacts.get("linkedin", ""))
        contacts["notion"] = st.text_input("Notion", value=contacts.get("notion", ""))
        contacts["resume_pdf"] = st.text_input("Resume PDF URL", value=contacts.get("resume_pdf", ""))

    with tab_projects:
        project_frame = projects_to_frame(editable.get("projects", []))
        edited_projects = st.data_editor(project_frame, num_rows="dynamic", use_container_width=True, height=420)
        editable["projects"] = normalize_project_rows(edited_projects)

    with tab_research:
        research = editable.setdefault("research", {})
        research["title"] = st.text_input("Research title", value=research.get("title", ""))
        research["summary"] = st.text_area("석사 연구 요약", value=research.get("summary", ""), height=120)
        research["problem_definition"] = st.text_area("문제 정의", value=research.get("problem_definition", ""), height=120)
        research["proposed_method"] = st.text_area("제안 방법", value=research.get("proposed_method", ""), height=120)
        research["experimental_results"] = st.text_area("실험 결과", value=research.get("experimental_results", ""), height=120)
        research["limitations_and_extensions"] = st.text_area("한계 및 확장 방향", value=research.get("limitations_and_extensions", ""), height=120)

    with tab_blog:
        blog_frame = pd.DataFrame(editable.get("blog", []))
        edited_blog = st.data_editor(blog_frame, num_rows="dynamic", use_container_width=True, height=420)
        editable["blog"] = normalize_blog_rows(edited_blog)

    with tab_json:
        raw_json = st.text_area("전체 콘텐츠 JSON", value=json.dumps(editable, indent=2, ensure_ascii=False), height=520)
        if st.button("Raw JSON 저장", type="secondary"):
            try:
                parsed = json.loads(raw_json)
                if not isinstance(parsed, dict):
                    raise ValueError("최상위 JSON은 object여야 합니다.")
                save_content(parsed)
                st.success("Raw JSON을 저장했습니다.")
                st.rerun()
            except json.JSONDecodeError as exc:
                st.error(f"JSON 형식 오류: {exc}")
            except ValueError as exc:
                st.error(str(exc))

    if st.button("Save changes", type="primary"):
        save_content(editable)
        st.success("저장했습니다.")
        st.rerun()

    if st.button("Logout"):
        st.session_state["portfolio_admin_authenticated"] = False
        st.rerun()


def main() -> None:
    """Run portfolio site."""
    st.set_page_config(page_title="AI Engineering Portfolio", page_icon="AI", layout="wide")
    content = load_content()
    pages = {
        "Home": page_home,
        "Projects": page_projects,
        "Research": page_research,
        "Blog": page_blog,
        "Resume": page_resume,
        "Admin": page_admin,
    }
    with st.sidebar:
        selected = st.radio("Navigation", list(pages), index=0)
    pages[selected](content)


if __name__ == "__main__":
    main()
