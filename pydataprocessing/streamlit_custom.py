import streamlit as st
from st_pages import Page, Section, show_pages
sss = st.session_state

from os.path import join

def v_space(height, sb=False) -> None:
    """
    Creates vertical spacing by writing newline characters.
    Parameters:
        height (int): The number of newline characters to write.
        sb (bool, optional): Whether to write the newline characters to the sidebar or not. Defaults to False.
    Returns:
        None
    """
    for _ in range(height):
        if sb:
            st.sidebar.write("\n")
        else:
            st.write("\n")


def file_nullify():
    st.session_state.current_uploaded_files_names = list()

def error(body, icon):
    st.error(
        body=body,
        icon=icon)
    st.stop()


def set_pages_layout(icon):
    st.set_page_config(
        page_icon=icon,
        layout="centered",
        initial_sidebar_state="expanded",
    )

    home_path = r"C:\Users\ZhironkinAA\PycharmProjects\pythonProject3\!scripts\one_page_app\Home.py"
    home_dir = r"C:\Users\ZhironkinAA\PycharmProjects\pythonProject3\!scripts\one_page_app"
    pages_dir = r"C:\Users\ZhironkinAA\PycharmProjects\pythonProject3\!scripts\one_page_app\pages"
    
    
    sss["home_path"] = home_path
    sss["home_dir"] = home_dir
    sss["pages_dir"] = pages_dir
    
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "Сервисы социологии";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;

            }

            [data-testid="stSidebarNav"]::after {
                content: "Сервисы таргета";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: -90px;

            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    show_pages(
        [
            Page(home_path, "Домашняя страница", ":house:"),
            Page(join(pages_dir, "weights", "wt_cl.py"), "Чистка и взвешивание", ":scales:"),
            Page(join(pages_dir, "table_worker", "table_worker.py"), "Таблицы", "📅"),
            Page(join(pages_dir, "ppt_gen", "ppt_gen.py"), "Франкенштейн отчета", "🧟‍♂️"),
            Page(join(pages_dir, "blocks", "blocks.py"), "Блоки", "🧱"),

            Section(" ", " "),

            Page(join(pages_dir, "target_id_converter.py"), "Конвертирование айдишек", "🔑"),
        ]
    )


def clear_sss():
    print("clear session_state")
    for k in st.session_state:
        if k not in ["home_path", "home_dir", "pages_dir"]:
            del st.session_state[k]