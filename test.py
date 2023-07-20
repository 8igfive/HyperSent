import os
import openai
openai.api_key = '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'

messages_first = [
    {"role": "system", "content": 
'''You are a helpful assistant. When the user enters a sentence, generate a new sentence that is semantically similar to the input sentence. The following are examples of semantically similar sentences.

- Example 1:
  - Sentence 1: U.S. prosecutors have arrested more than 130 individuals and have seized more than $17 million in a continuing crackdown on Internet fraud and abuse.
  - Sentence 2: More than 130 people have been arrested and $17 million worth of property seized in an Internet fraud sweep announced Friday by three U.S. government agencies.
- Example 2:
  - Sentence 1: The hearing occurred a day after the Pentagon for the first time singled out an officer, Dallager, for not addressing the scandal.
  - Sentence 2: The hearing came one day after the Pentagon for the first time singled out an officer - Dallager - for failing to address the scandal.

Generate the semantically similar sentence directly without any other interpretation.'''},
# Try to make the generated sentences have different grammar and vocabulary than the input sentences. Give the results directly without any other interpretation.
    {"role": "user", "content": "It succeeded the Carrero Blanco government and was the Government of Spain from 4 January 1974 to 12 December 1975, a total of days, or ."}
]

messages_level = [
    {"role": "system", "content": 
'''You are a helpful assistant. Your task is to generate a new sentence based on the user's input sentence by removing certain information. The examples below illustrate this process:

- Example 1: Sentence 2 is created by removing information from Sentence 1.
  - Sentence 1: I would urge the French Presidency to do their utmost to overcome this divide, because it is extremely destructive.
  - Sentence 2: I make an urgent appeal to the Presidency to do everything in its power to fill this role.
- Example 2: Sentence 2 is created by removing information from Sentence 1.
  - Sentence 1: Those who stand apart from reinforced cooperation are really being given the same freedom as the back seat passengers in a car.
  - Sentence 2: Those who refuse and enjoy the same freedom of movement as the passengers out of a car.

Generate the new sentence directly without any other interpretation.'''},
# Try to make the generated sentences have different grammar and vocabulary than the input sentences. Give the results directly without any other interpretation.
    {"role": "user", "content": "It succeeded the Carrero Blanco government and was the Government of Spain from 4 January 1974 to 12 December 1975, a total of days, or ."}
]
# [
#     {"role": "system", "content": 
# '''You are a helpful assistant. When the user enters 3 sentences, generate a new sentence that has lower semantic similarity with the third input sentence compared to the semantic similarity between the first and second input sentences. The following examples demonstrate the semantic similarity comparison between different pairs of sentences:

# - Example 1: The semantic similarity of sentence pair 1 is higher than that of sentence pair 2.
#   - Sentence pair 1:
#     - Sentence 1: The Bush administration blames Hussein loyalists and foreign Muslim militants who have entered Iraq to fight U.S. troops for the wave of bombings and guerrilla attacks.
#     - Sentence 2: The Bush administration blames the wave of bombings and guerrilla attacks on Saddam loyalists and foreign Muslim militants who have entered Iraq to fight U.S. troops.
#   - Sentence pair 2:
#     - Sentence 1: Mr President, I would like to focus my speech on Mrs Lalumière's report, which I think is clearly worded and well put together.
#     - Sentence 2: Mr President, I shall concentrate on the report by Mrs Lalumière in my speech, a report that I find well-written and clear.
# - Example 2: The semantic similarity of sentence pair 1 is higher than that of sentence pair 2.
#   - Sentence pair 1:
#     - Sentence 1: Ballmer has been vocal in the past warning that Linux is a threat to Microsoft.
#     - Sentence 2: In the memo, Ballmer reiterated the open-source threat to Microsoft.
#   - Sentence pair 2:
#     - Sentence 1: The Dow Jones Industrial Average fell 0.7 per cent to 9,547.43 while the S&P 500 was 0.8 per cent weaker at 1,025.79.
#     - Sentence 2: The Dow Jones industrial average fell 44 points, or 0.46 percent, to 9,568.

# Generate the new sentence by removing some information in the third sentence and make sure that the semantic similarity between the new sentence and the third input sentence is lower than that between the first and second input sentences. Give the result directly without any other interpretation.'''},
#     {"role": "user", "content": 
# '''Sentence 1: 
# It succeeded the Carrero Blanco government and was the Government of Spain from 4 January 1974 to 12 December 1975, a total of days, or .
# Sentence 2: 
# It took over the Carrero Blanco administration and served as the leadership of Spain from January 4, 1974 to December 12, 1975, spanning several days or .
# Sentence 3:
# It succeeded the Carrero Blanco government and was the Government of Spain from 4 January 1974 to 12 December 1975, a total of days, or .'''}
# ]


messages_casual = [
    {"role": "system", "content": 
'''You are a helpful assistant.'''},
# Try to make the generated sentences have different grammar and vocabulary than the input sentences. Give the results directly without any other interpretation.
    {"role": "user", "content": 
'''帮我修改以下用英文写的任务描述，使其表述更加明确：
```Your task is to generate a new sentence that conveys essentially the same information as the user's input sentence, but differs in the manner of expression or the vocabulary used.```'''
    }
]

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=messages_casual
)

print(completion.choices[0].message['content'])