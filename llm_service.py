import requests

def get_llm_advice(user_text):
    prompt = """
    Act as an ai health assistant and assist the user \n:
    {user_text}
    Give advices and warn of symptoms are severe and dangeous.
    """
    
    