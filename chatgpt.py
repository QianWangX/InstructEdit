import os
import openai
import re



openai.api_key = ''

TASK_PROMPT = 'There is a system that can edit images by the user instructions.'\
'There are three components in the system: a segmentation model, an image editing model, and a language model, which is you.'\
'The segmentation model and the image editing model need you to parse the user instructions and input the text prompt they need.'\
    'The first one is a segmentation model which will segment an image by the text keyword.'\
        'The second one is an image editing model, it needs two prompts. The first prompt describe the content user want to edit, and the second prompt describe the object or attribute after editing.'\
            'The user will give you an instruction, and you will give the two models the prompt they need.'\
                'For example, if the user says \'Change the dog to a cat\','\
                    'Then you need to give the segmentation model only the keyword \'Dog\','\
                        'Then you need to give the image editing model two text prompts: \'Photo of a dog\', and \'Photo of a cat\'.'\
                        'Your answer should be in the form of \'Segmentation prompt: Dog. Editing prompt 1: \'Photo of a dog\'. Editing prompt2: \'Photo of a cat\'\'.'\
                'For example, if the user says \'Add an glasses to the boy\','\
                    'Then you need to give the segmentation model only the keyword \'Boy\','\
                        'Then you need to give the image editing model two text prompts: \'Photo of a boy\', and \'Photo of a boy wearing glasses\'.'\
                        'Your answer should be in the form of \'Segmentation prompt: Boy. Editing prompt 1: \'Photo of a boy\'. Editing prompt2: \'Photo of a boy wearing glasses\'\'.'\
                'For example, if the user says \'Remove the box and phone on the table\','\
                    'Then you need to give the segmentation model only the keywords \'box\' and \'phone\' but not \'table\','\
                        'Then you need to give the image editing model two text prompts: \'Photo of a box and a phone on the table\', and \'Photo of an empty and clean table\'.'\
                        'Your answer should be in the form of \'Segmentation prompt: box, phone. Editing prompt 1: \'Photo of a box and a phone on the table\'. Editing prompt2: \'Photo of an empty and clean table\'\'.' 
                        
TASK_PROMPT_1 = 'There is a system that can edit images by the user instructions.'\
'There are four components in the system: a segmentation model, an image editing model, a multimodal model and a language model, which is you.'\
'The segmentation model and the image editing model need you to parse the user instructions and input the text prompt they need. The multimodal model will descibe the input image, which will assist you provide the text prompt.'\
    'The segmentation model will segment an image by the text keyword.'\
        'The image editing model needs two prompts. The first prompt describe the content user want to edit, and the second prompt describe the object or attribute after editing.'\
            'The multimodal model will describe the content of the input image. You need to adjust the description to based on the user instructions to give segmentation model and the image editing model the editing prompts.'\
                'For example, if the user says \'Change the yellow dog to a cat\', and the multimodal model says \'Photo of a yellow dog and a white dog.\''\
                    'Then you need to give the segmentation model only the keyword \'Yellow dog\','\
                        'Then you need to give the image editing model two text prompts: \'Photo of a yellow dog and a white dog\', and \'Photo of a cat and a white dog\'.'\
                        'Your answer should be in the form of \'Segmentation prompt: Yellow, dog. Editing prompt 1: \'Photo of a yellow dog and a white dog\'. Editing prompt2: \'Photo of a cat and a white dog\'\'.'\
                'For example, if the user says \'Add an glasses to the boy\', and the multimodal model says \'Photo of a boy in the park.\''\
                    'Then you need to give the segmentation model only the keyword \'Boy\','\
                        'Then you need to give the image editing model two text prompts: \'Photo of a boy in the park.\', and \'Photo of a boy wearing glasses in the park.\'.'\
                        'Your answer should be in the form of \'Segmentation prompt: Boy. Editing prompt 1: \'Photo of a boy in the park.\'. Editing prompt2: \'Photo of a boy wearing glasses in the park.\'\'.'\
                'For example, if the user says \'Remove the box and phone\', and the multimodal model says \'Painting of a box and a phone on the table.\''\
                    'Then you need to give the segmentation model only the keywords \'box\' and \'phone\' but not \'table\','\
                        'Then you need to give the image editing model two text prompts: \'Photo of a box and a phone on the table\', and \'Photo of an empty and clean table\'.'\
                        'Your answer should be in the form of \'Segmentation prompt: box, phone. Editing prompt 1: \'Photo of a box and a phone on the table\'. Editing prompt2: \'Photo of an empty and clean table\'\'.'\
                            'Avoid using \'instead of\' in the Editing prompt 2.'\
                                'You can modify the description of the multimodal model to make the editing prompt more natural. Sometimes the description of the multimodal model is too lengthy.'\
                                    'For example, it repeats the objects for too many times. You can delete the repeated objects but do not delete other important description!.'                              

ITERATIVE_EDITING_PROMPT = 'Assume there is an image editing model and there is a parameter called \'encoding ratio\' in it.'\
    'It controls the degree of editing. The higher it is, the more the edited image will have stronger editing effect.'\
    'Encoding ratio is from 0 to 1.'\
'If the users are not satisfied with the result, they can give you another instruction to edit the image again.'\
    'For example, if the user instruction is \'Change the dog to a cat\', and the user feedback is \'The image looks too much like the cat\', or \'I want the image to look more like the dog\''\
        'Then it means the user wants stronger editing effect. So the encoding ratio should be higher. You can pick a value between current value and 0, for example 0.3.'\
            'Your answer should be only in the form of \'Encoding ratio: 0.3\'.'\
                'Please explicitly state the encoding ratio you want to set.'
            
            
def prepare_chatgpt_message(task_prompt, instructions, descriptions=None, feedbacks=None, sub_prompt=None):
    messages = [{"role": "system", "content": task_prompt}]
    
    if descriptions:
        assert len(descriptions) == len(instructions)
        for instruction, desciption in zip(instructions, descriptions):
            messages.append({'role': 'user', 'content': 'User instruction: {}'.format(instruction)})
            messages.append({'role': 'user', 'content': 'Multimodal description: {}'.format(desciption)})
    elif feedbacks:
        assert len(feedbacks) == len(instructions)
        for instruction, feedback in zip(instructions, feedbacks):
            messages.append({'role': 'user', 'content': 'User instruction: {}'.format(instruction)})
            messages.append({'role': 'user', 'content': 'User feedback: {}'.format(feedback)})
    else:
        for instruction in instructions:
            messages.append({'role': 'user', 'content': 'User instruction: {}'.format(instruction)})
    
    if sub_prompt:
        messages.append({"role": "system", "content": sub_prompt})
    
    return messages


def get_reply_from_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def parse_chatgpt_reply(reply): 
    # before editing there can be a space, no space, a dot, a comma or a linebreak.
    seg_prompt = re.search(r'Segmentation prompt: (.*)[\.]*[\s\r\n]*Editing prompt[\s\r\n]*1', reply).group(1)
    edit_prompt1 = re.search(r'Editing prompt[\s\r\n]*1:[\s\r\n]*(.*)[\.]*[\s\r\n]*Editing prompt[\s\r\n]*2', reply).group(1)
    edit_prompt2 = re.search(r'Editing prompt[\s\r\n]*2:[\s\r\n]*(.*)[\.]*', reply).group(1)
    return seg_prompt, edit_prompt1, edit_prompt2

def parse_chatgpt_reply_encoding_ratio(reply): 
    # encoding ratio is a float number between 0 and 1.
    # parse the encoding ratio which is a float number between 0 and 1.
    encoding_ratio = re.search(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?', reply).group(1)
    encoding_ratio = float(encoding_ratio)
    
    return encoding_ratio

def call_chatgpt(instructions, descriptions=None, feedbacks=None):
    if descriptions:
        chatgpt_messages = prepare_chatgpt_message(TASK_PROMPT_1, instructions, descriptions=descriptions)
    elif feedbacks:
        chatgpt_messages = prepare_chatgpt_message(ITERATIVE_EDITING_PROMPT, instructions, feedbacks=feedbacks)
    else:
        chatgpt_messages = prepare_chatgpt_message(TASK_PROMPT, instructions)
        
    reply, total_tokens = get_reply_from_chatgpt(chatgpt_messages)
    print(reply)
    
    if feedbacks:
        encoding_ratio = parse_chatgpt_reply_encoding_ratio(reply)
        return encoding_ratio
    else:
        seg_prompt, edit_prompt1, edit_prompt2 = parse_chatgpt_reply(reply)
        return seg_prompt, edit_prompt1, edit_prompt2, total_tokens

