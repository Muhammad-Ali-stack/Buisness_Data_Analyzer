import plotly.express as px
import streamlit as st

def visualize_data(df):
    st.subheader("📊 Visualizations")
    numeric_cols = df.select_dtypes(include='number').columns

    if "date" in df.columns and len(numeric_cols) > 0:
        x_col = "date"
        y_col = numeric_cols[0]
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} Over Time")
        st.plotly_chart(fig)

    if len(numeric_cols) > 1:
        fig2 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Correlation Plot")
        st.plotly_chart(fig2)
