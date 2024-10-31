def get_chatbot_response(client, model_name, messages):
    """
    Get response from chat completion API with error handling
    """
    try:
        # Validate inputs
        if not client or not model_name or not messages:
            print("Error: Missing required parameters")
            return None

        input_messages = []
        for message in messages:
            input_messages.append({"role": message["role"], "content": message["content"]})

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=input_messages,
                temperature=0,
                top_p=0.8,
                max_tokens=2000,
            )
            
            # Print response for debugging
            print(f"API Response: {response}")
            
            if not response or not response.choices:
                print("Error: Empty response received")
                return None
                
            return response.choices[0].message.content

        except Exception as e:
            print(f"API Error: {str(e)}")
            return None

    except Exception as e:
        print(f"Error in get_chatbot_response: {str(e)}")
        return None

def get_embedding(embedding_client, model_name, text_input):
    try:
        if not embedding_client or not model_name or not text_input:
            print("Error: Missing required parameters for embedding")
            return None

        output = embedding_client.embeddings.create(input=text_input, model=model_name)
        
        embeddings = []
        for embedding_object in output.data:
            embeddings.append(embedding_object.embedding)

        return embeddings

    except Exception as e:
        print(f"Error in get_embedding: {str(e)}")
        return None

def double_check_json_output(client, model_name, json_string):
    try:
        if not client or not model_name or not json_string:
            print("Error: Missing required parameters for JSON check")
            return None

        prompt = f"""You will check this json string and correct any mistakes that will make it invalid. Then you will return the corrected json string. Nothing else. 
        If the Json is correct just return it.

        Do NOT return a single letter outside of the json string.

        {json_string}
        """

        messages = [{"role": "user", "content": prompt}]

        response = get_chatbot_response(client, model_name, messages)
        if not response:
            print("Error: Failed to get response from chatbot")
            return None

        return response

    except Exception as e:
        print(f"Error in double_check_json_output: {str(e)}")
        return None