import streamlit as st
from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub
load_dotenv()
import os
os.environ["HUGGINGFACE_API_KEY"]=os.getenv("HUGGINGFACE_API_KEY")