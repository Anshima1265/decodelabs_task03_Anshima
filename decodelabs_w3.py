import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datetime import datetime

courses = {
    "Course": [
        "Python Programming",
        "Machine Learning",
        "Artificial Intelligence",
        "Data Science",
        "Deep Learning",
        "Web Development",
        "Cyber Security",
        "Cloud Computing",
        "Android Development",
        "Game Development",
        "UI UX Design",
        "Blockchain",
        "DevOps",
        "Computer Vision",
        "Natural Language Processing",
        "C++ Programming",
        "Java Programming",
        "React Development"
    ],

    "Skills": [
        "python coding programming software",
        "python machine learning algorithms data",
        "artificial intelligence ai machine learning automation",
        "data science analytics statistics python visualization",
        "neural networks ai deep learning",
        "html css javascript frontend backend web development",
        "cyber security networking ethical hacking",
        "aws azure cloud devops infrastructure",
        "android java kotlin mobile development",
        "unity gaming graphics programming game development",
        "figma ui ux design creativity",
        "blockchain cryptocurrency smart contracts",
        "docker kubernetes automation cloud devops",
        "computer vision image processing ai",
        "nlp chatbot text processing ai",
        "c++ programming oop dsa competitive coding",
        "java programming spring boot backend",
        "react javascript frontend web development"
    ]
}

df = pd.DataFrame(courses)

available_interests = [
    "python",
    "machine learning",
    "ai",
    "artificial intelligence",
    "data science",
    "deep learning",
    "web development",
    "cyber security",
    "cloud",
    "android",
    "game development",
    "ui ux",
    "blockchain",
    "devops",
    "computer vision",
    "nlp",
    "c++",
    "java",
    "react"
]

print("=" * 75)
print("AI BASED CAREER RECOMMENDATION SYSTEM")
print("=" * 75)

name = input("Enter Your Name: ")

print("\nAvailable Interests:\n")

for i, interest in enumerate(available_interests, start=1):
    print(f"{i}. {interest.title()}")

user_input = input(
    "\n\nEnter your interests separated by commas: "
).lower()

user_interests = [x.strip() for x in user_input.split(",")]

valid_interests = []
invalid_interests = []

for interest in user_interests:
    if interest in available_interests:
        valid_interests.append(interest)
    else:
        invalid_interests.append(interest)

if len(valid_interests) == 0:
    print("\nNo valid interests found.")
    print("\nPlease choose from the available interests only.")
    exit()

if len(invalid_interests) > 0:
    print("\nIgnored Invalid Interests:")
    for item in invalid_interests:
        print("-", item)

user_profile = " ".join(valid_interests)

vectorizer = TfidfVectorizer()

all_text = df["Skills"].tolist()
all_text.append(user_profile)

vectors = vectorizer.fit_transform(all_text)

course_vectors = vectors[:-1]
user_vector = vectors[-1]

scores = cosine_similarity(
    user_vector,
    course_vectors
).flatten()

df["Similarity Score"] = scores * 100

recommendations = df.sort_values(
    by="Similarity Score",
    ascending=False
)

top5 = recommendations.head(5)

print("\n")
print("=" * 75)
print(f"TOP RECOMMENDATIONS FOR {name.upper()}")
print("=" * 75)

for i, (_, row) in enumerate(top5.iterrows(), start=1):
    print(
        f"{i}. {row['Course']} --> {row['Similarity Score']:.2f}% Match"
    )

best_course = top5.iloc[0]["Course"]
best_score = top5.iloc[0]["Similarity Score"]

print("\n")
print("=" * 75)
print("AI ANALYSIS")
print("=" * 75)

print(f"Best Career Match      : {best_course}")
print(f"Confidence Score       : {best_score:.2f}%")
print(f"Interests Considered   : {', '.join(valid_interests)}")

history = pd.DataFrame({
    "Name": [name],
    "Interests": [", ".join(valid_interests)],
    "Best Recommendation": [best_course],
    "Score": [round(best_score, 2)],
    "Date": [datetime.now()]
})

try:
    old_history = pd.read_csv(
        "recommendation_history.csv"
    )
    history = pd.concat(
        [old_history, history],
        ignore_index=True
    )
except:
    pass

history.to_csv(
    "recommendation_history.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.bar(
    top5["Course"],
    top5["Similarity Score"]
)

plt.title("Top 5 Career Recommendations")
plt.xlabel("Career Domains")
plt.ylabel("Similarity Score (%)")
plt.xticks(rotation=25)
plt.tight_layout()

plt.show()

plt.figure(figsize=(8, 8))

plt.pie(
    top5["Similarity Score"],
    labels=top5["Course"],
    autopct="%1.1f%%"
)

plt.title("Recommendation Distribution")

plt.show()

print("\nRecommendation history saved successfully.")
