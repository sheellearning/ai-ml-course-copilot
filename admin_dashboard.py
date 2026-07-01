import streamlit as st

from core.usage_logger import get_metrics


st.set_page_config(page_title="Admin Dashboard", layout="wide")

st.title("RAG System Monitoring")


# Load usage statistics from the saved request logs.
metrics = get_metrics()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Requests", metrics["total_requests"])

with col2:
    st.metric("Success Rate", f"{metrics['success_rate']}%")

with col3:
    st.metric("No Answer Count", metrics["failed_answers"])


st.divider()
st.subheader("Most Asked Questions")

if metrics["most_asked"]:
    for question, count in metrics["most_asked"]:
        st.write(f"{question} ({count} times)")
else:
    st.write("No questions logged yet.")


st.divider()
st.subheader("Recent Requests")

logs = metrics["logs"]

if logs:
    # Display the 20 most recent usage records.
    st.dataframe(logs[-20:], use_container_width=True)
else:
    st.write("No requests logged yet.")