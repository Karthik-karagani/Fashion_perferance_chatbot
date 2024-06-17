import gradio as gr
import pandas as pd
from openai import OpenAI
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
import os
from typing import Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv('.env')

final_data = pd.read_csv('processed_csv.csv')

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
)

sim_model = SentenceTransformer("all-MiniLM-L6-v2")

pinecone_index_name = 'fashion-chatbopt-index'
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index = pc.Index(pinecone_index_name)

def get_query_filter(category,colour,gender,max_price,min_price):
    query_filter = {}
    if category:
        query_filter["category"] =  { "$eq": category.lower() }

    if colour and colour!='skin colour':
        query_filter["color"] =  { "$eq": colour.lower() }

    if gender:
        query_filter["gender"] =  { "$eq": gender.lower() }

    if max_price and min_price:
        query_filter["$and"] = [{"price": { "$gte": min_price }}, {"price": { "$lte": max_price }}]
    else:
        if max_price:
            query_filter["price"] =  { "$lte": max_price }
        if min_price:
            query_filter["price"] =  { "$gte": min_price }

    return query_filter

def get_similar_prod_details(query,category,colour,gender,max_price,min_price,k=3):
    query_filter = get_query_filter(category,colour,gender,max_price,min_price)
    query_vec = sim_model.encode(query).tolist()
    
    query_response = index.query(
        top_k=k,
        include_metadata=True,
        vector=query_vec,
        filter=query_filter)
    
    results = [i['metadata'] for i in query_response['matches']]
    return results

def get_similar_product_df(query,category,colour,gender,max_price,min_price,k=3):
    filter_data = pd.read_csv('final_dataset.csv')
    
    if category:
        tmp = filter_data[filter_data['category'].str.lower().str.contains(category.lower())]
        if not tmp.empty:
            filter_data = tmp
        
    if colour:
        tmp = filter_data.dropna()
        tmp = tmp[tmp['colour'].str.lower().str.contains(colour.lower())]
        if not tmp.empty:
            filter_data = tmp
            
    if gender:
        tmp = filter_data[filter_data['gender'].str.lower().str.contains(gender.lower())]
        if not tmp.empty:
            filter_data = tmp
            
    if max_price:
        tmp = filter_data[filter_data['price'] <= max_price]
        if not tmp.empty:
            filter_data = tmp
            
    if min_price:
        tmp = filter_data[filter_data['price'] >= min_price]
        if not tmp.empty:
            filter_data = tmp
           
    if not filter_data.empty:
        return filter_data.head(k).to_dict(orient='index')
    else:
        return {}



class FashionParamsInput(BaseModel):
    descp: str = Field(..., description="description of the product to search for")
    colour: str = Field(..., description="colour of the product to search for")
    gender: str = Field(..., description="gender of the product to search for")
    category: str = Field(..., description="category of the product to search for")
    min_price: float = Field(..., description="minimum price of the product to search for")
    max_price: float = Field(..., description="maximum price of the product to search for")

@tool(args_schema=FashionParamsInput)
def get_top_similar_products(descp: str, colour: str, gender: str, category: str, min_price: float, max_price: float) -> list:
    """Search top similar product based on title,brand and price"""
    r = get_similar_product_df(descp,category,colour,gender,max_price,min_price,k=3)
    
    return r

tools = [get_top_similar_products]

functions = [convert_to_openai_function(f) for f in tools]
model = ChatOpenAI(temperature=0).bind(functions=functions)

questions_list = [ 'what category you looking for?',
                  'what colour you looking for?',
                  'what gender are you interested in?',
                  
                  'what is min and max price range you are looking for?']

system_prompt = "As an eCommerce chatbot, your objective is to assist users by asking a series of questions one by one given below to understand their preferences. Based on their responses, you will then proceed to identify the most suitable product or service that aligns with their tastes and needs."
system_prompt += '\n'+'###'*30
questions_to_ask = '\n'.join(questions_list)
system_prompt += f'\nQuestion:\n{questions_to_ask}\n'

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | prompt | model | OpenAIFunctionsAgentOutputParser()


memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, memory=memory, verbose=True)

class Chat:

    def __init__(self, system: Optional[str] = None):
        self.system = system
        self.messages = []

        if system is not None:
            self.messages.append({
                "role": "system",
                "content": system
            })

    def prompt(self, content: str) -> str:
          self.messages.append({
              "role": "user",
              "content": content
          })
          query = self.messages[-1]['content']
          response = agent_executor.invoke({"input": query})
          response_content = response['output']
          
          self.messages.append({
              "role": "assistant",
              "content": response_content
          })
          return response_content
      
      
chat = Chat(system="You are a helpful assistant.")
df = pd.read_csv('colours.csv')
colors_list = df['skintone type'].unique().tolist()

def respond(message, chat_history):
    bot_message = chat.prompt(content=message)
    chat_history.append((message, bot_message))
    return "", chat_history

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)

skin_tone_urls = {
    'fair skin': 'https://ih1.redbubble.net/image.1097628469.7960/flat,750x,075,f-pad,750x1000,f8f8f8.jpg',
    'light skin': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjyEX0WszwY4OkocedmolcT_BvskpseiK9Yp-OAgArhZ610YpvbA4nYBOch3fG0AtMfJE&usqp=CAU',
    'medium skin': 'https://garden.spoonflower.com/c/10197941/p/f/m/BtdGZR1QE553JwYd85VUN5S-L851ueu2knyqBdXGqohpCzW016gp/Skin%20Tone%20Complexion%2033.jpg',
    'dark skin': 'https://garden.spoonflower.com/c/13719863/p/f/m/Ew6bvJiKk9Vn2hS6a8znvkTe2lPfT9rlekTXUag4pVOtLcVMpWyomsM/Complexion%20-%20Irish%20Coffee%20-%2023%2F28%20-%20Very%20Dark.jpg'
}






css = """


.heading-container {
    text-align: center;
    margin-bottom: 20px;
}

.heading {
    display: inline-block;
    font-weight: bold;
    border: 2px solid black;
    padding: 10px;
    font-size: 24px; /* Adjust the font size as needed */
    background-color: black; /* Add background color */
    border-radius: 10px; /* Add border radius */
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div class='heading-container'>
            <h1 class='heading'>Fashion Preference Chatbot</h1>
        </div>
    """)
    with gr.Row():
        vid = gr.Video(sources=['webcam'],height=None,width=None)
        with gr.Column():
            ch = gr.Dropdown(choices=colors_list,value=0,label='Choose your skin tone')
            submit_btn = gr.Button(value='submit')
            img2 = gr.Image(value=None,height=None,width=None)
    img = gr.HTML(value=None)
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def get_images(ch):
        if ch!=None:
            sample = df[df['skintone type']==ch][['colour','image url']]
            d = dict(zip(sample['colour'].values,sample['image url'].values))
            html_content = '<div style="display: flex;">'
            for color, image_url in d.items():
                html_content += f'<div style="margin: 10px; text-align: center;">'
                html_content += f'<img src="{image_url}" alt="{color}" style="width: 200px; height:200 ;">'
                html_content += f'<p>{color}</p>'
                html_content += f'</div>'
            html_content += '</div>'
            return html_content,gr.Image(value=skin_tone_urls[ch],height=300 ,width=None)

    submit_btn.click(get_images, [ch], [img,img2])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    chatbot.like(vote, None, None)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
