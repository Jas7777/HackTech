# HackTech

#
Data collected form Kaggle- an open sourse data platform, full data:https://www.kaggle.com/datasets/cihan063/autism-image-data

## Inspiration
I was inspired to make a difference, after going to rural areas of Southeastern Asia, and learning about the struggle of ASD parents in these areas, in terms of economic and access.
## What it does
Autism spectrum disorder(ASD) is a neurological disorder caused by differences in the brain. The most common way to detect autism in children is ADI-R. It helps doctors and their patients detect autism and begin treatment. This effective method can cost upwards of 5,000 USD for families. This is more than an entire month's salary to most people in developing countries. To reduce the financial burden while still having the same benefits as an ADI-R, I plan to create an AI model that can detect autism early in children.  
## How we built it
The backend of the code was developed using python, and the sklearn library. The frontend end was developing using the streamlit library.
## Challenges we ran into
Initially I didn't know which sklearn model was better, so I trusted the model that performed better in the last few rounds. Each mistake divides the trust for the model into A and B.  “A” happens when I have the error of the model predicting "Autistic" instead of "Non-Autistic". “B” happens when I  have the "Non-Autistic "  instead of "Autistic". If i believe A is less important than error B, I set error B to have a higher penalty score. Then I finalized the weights depending on the weight error. In this case, having the model predict autism in a non-autistic person is better than an autistic person as a non autistic. By adopting this strategy, I can choose the most accurate model.
## Accomplishments that we're proud of
Learning the Streamlit platform was a new hurdle for me to overcome. This was also my first project where I coded both the front and backend.
## What we learned
I learned how to use machine learning algorithms, and deploy them on a website.
## What's next for AI+Autism Early Diagnosis tool 
I hope to add other features that can determine if an Individual has Autism like, writing in the form of a text or conversation.  
