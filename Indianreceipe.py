import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to generate Indian recipes using LLama 2 model
def generateRecipe(input_text, cuisine_type):
    # LLama2 model initialization
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 200,
                                'temperature': 0.8})

    # Prompt Template for recipes
    template = """
You are an Indian chef. Provide a simple recipe for a dish mentioned in {input_text} using the style {cuisine_type}
"""


    prompt = PromptTemplate(input_variables=["cuisine_type", "input_text"],
                            template=template)

    # Generate response from the LLama 2 model
    response = llm(prompt.format(cuisine_type=cuisine_type, input_text=input_text))
    return response

# Streamlit UI
st.set_page_config(page_title="Indian Recipes Generator",
                   page_icon='🍛',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Indian Recipes 🍛")

input_text = st.text_input("Enter the Dish name")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    cuisine_type = st.selectbox('Select Cuisine Type',
                               ('Veg', 'Nonveg', 'Vegan'), index=0)

with col2:
    # Additional options specific to the cuisine type can be added here
    pass

submit = st.button("Generate Recipe")

# Display final response
if submit:
    response = generateRecipe(input_text, cuisine_type.lower())
    st.write(response)
