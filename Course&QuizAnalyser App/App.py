import google.generativeai as genai
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def configure_genai_api():
    """Configure Gemini API."""
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("API key not found. Please set the 'GOOGLE_API_KEY' environment variable.")
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

def generate_question(content, course, level):
    """Generate a single question based on course content and difficulty level."""
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("API key not configured. Please set the 'GOOGLE_API_KEY' environment variable.")
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Generate a {level} difficulty question for the course '{course}' based on the following content: {content}"
        response = model.generate_content(prompt)
        question = response.text.strip()
        return question
    except Exception as e:
        print(f"Error generating {level} question: {e}")
        return None

def evaluate_answer(course, question, user_answer):
    """Evaluate the user's answer using Gemini API."""
    try:
        if not GOOGLE_API_KEY:
            raise ValueError("API key not configured. Please set the 'GOOGLE_API_KEY' environment variable.")
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Question: {question}\nUser Answer: {user_answer}\nIs the user's answer correct? Respond with 'Yes' or 'No'."
        response = model.generate_content(prompt)
        evaluation = response.text.strip().lower()
        if 'yes' in evaluation:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return False

def quiz_user(df, course, content):
    """Implement adaptive quiz logic."""
    level = 'easy'  
    max_questions = 5
    asked_questions = 0
    correct_answers = 0

    while asked_questions < max_questions:
        question = generate_question(content, course, level)
        if not question:
            print("Unable to generate question. Ending quiz.")
            break

        print(f"\nQuestion {asked_questions + 1} ({level.capitalize()}):")
        print(question)
        user_answer = input("Your Answer: ")
        is_correct = evaluate_answer(course, question, user_answer)

        if is_correct:
            print("Correct!")
            correct_answers += 1
         
            if level == 'easy':
                level = 'medium'
            elif level == 'medium':
                level = 'hard'
            elif level == 'hard':
                level = 'expert'
        else:
            print("Incorrect!")
          
            if level == 'expert':
                level = 'hard'
            elif level == 'hard':
                level = 'medium'
            elif level == 'medium':
                level = 'easy'
        
        asked_questions += 1

    print(f"\nQuiz completed! You answered {correct_answers} out of {max_questions} questions correctly.")

def generate_course_quiz(df, course):
    """Generate and conduct a quiz for a specific course."""
    print(f"\nStarting quiz for course: {course}")
    course_descriptions = {
        'Python Programming': 'Learn the basics of Python programming including syntax, variables, and data types.',
        'Data Science': 'Introduction to data science concepts including data analysis, visualization, and machine learning.',
        'Machine Learning': 'Understand the fundamentals of machine learning algorithms and their applications.',
        'Web Development': 'Learn how to build websites using HTML, CSS, and JavaScript.',
        
    }
    content = course_descriptions.get(course, f"Details for {course}")
    if not content:
        print(f"No description found for course: {course}")
        return
    quiz_user(df, course, content)



def analyze_course_data(file_path):
    """Analyze student course data."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("Dataset is empty.")
        
        print("\nCourse Data Summary:")
        print(df.info())

      
        if df['CompletionStatus'].dtype == object:
            completion_mapping = {'Completed': 'Completed', 'In Progress': 'In Progress', 'Not Started': 'Not Started'}
            df['CompletionStatus'] = df['CompletionStatus'].map(completion_mapping)
        else:
            
            completion_mapping = {1: 'Completed', 0: 'Not Completed/In Progress'}
            df['CompletionStatus'] = df['CompletionStatus'].map(completion_mapping)

        completed = df[df['CompletionStatus'] == 'Completed']
        in_progress = df[df['CompletionStatus'] == 'In Progress']
        not_started = df[df['CompletionStatus'] == 'Not Started']

        print(f"\nCourses Completed: {len(completed)}")
        print(f"Courses In Progress: {len(in_progress)}")
        print(f"Courses Not Started: {len(not_started)}")
        return df
    except Exception as e:
        print(f"Error analyzing course data: {e}")
        return None

def visualize_course_data(df):
    """Visualize course completion statuses."""
    if df is not None:
        status_counts = df['CompletionStatus'].value_counts()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=status_counts.index, y=status_counts.values, palette='viridis')
        plt.title('Course Completion Statuses')
        plt.xlabel('Status')
        plt.ylabel('Count')
        plt.show()

 
        plot_completion_trend(df)
        plot_completion_duration(df)
        plot_status_by_category(df)
        plot_correlation_heatmap(df)
        plot_status_pie_chart(df)
        plot_completion_rates(df, 'Course')
        plot_scores_by_status(df)  

def plot_completion_trend(df):
    """Plot course completion trends over time."""
    if 'CompletionDate' in df.columns:
        df['CompletionDate'] = pd.to_datetime(df['CompletionDate'], errors='coerce')
        completion_trend = df[df['CompletionStatus'] == 'Completed'].groupby(df['CompletionDate'].dt.to_period('M')).size()
        plt.figure(figsize=(10, 6))
        completion_trend.plot(kind='line', marker='o', color='green')
        plt.title('Trend of Course Completions Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Completions')
        plt.grid(True)
        plt.show()
    else:
        print("CompletionDate column not found.")

def plot_completion_duration(df):
    """Visualize distribution of course completion durations."""
    if 'StartDate' in df.columns and 'CompletionDate' in df.columns:
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        df['CompletionDate'] = pd.to_datetime(df['CompletionDate'], errors='coerce')
        df['Duration'] = (df['CompletionDate'] - df['StartDate']).dt.days
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Duration'].dropna(), bins=20, kde=True, color='blue')
        plt.title('Distribution of Course Completion Durations')
        plt.xlabel('Days')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("StartDate or CompletionDate column not found.")

def plot_status_by_category(df):
    """Plot completion status by course category."""
    if 'Category' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='Category', hue='CompletionStatus', palette='coolwarm')
        plt.title('Completion Status by Course Category')
        plt.xlabel('Course Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Completion Status')
        plt.show()
    else:
        print("Category column not found.")

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()
    else:
        print("No numeric features to compute correlation.")

def plot_status_pie_chart(df):
    """Visualize completion statuses as a pie chart."""
    status_counts = df['CompletionStatus'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Completion Status Distribution')
    plt.show()

def plot_completion_rates(df, group_by_column):
    """Compare completion rates across groups."""
    if group_by_column in df.columns:
        if df['CompletionStatus'].dtype == object:
            df['CompletionStatus_numeric'] = df['CompletionStatus'].map({'Completed': 1, 'In Progress': 0, 'Not Started': 0})
        grouped = df.groupby(group_by_column)['CompletionStatus_numeric'].apply(lambda x: (x == 1).mean() * 100)
        plt.figure(figsize=(10, 6))
        grouped.sort_values(ascending=False).plot(kind='bar', color='purple')
        plt.title(f'Completion Rates by {group_by_column}')
        plt.xlabel(group_by_column)
        plt.ylabel('Completion Rate (%)')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"{group_by_column} column not found.")

def plot_scores_by_status(df):
    """Plot box plot of scores by completion status."""
    if 'Score' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='CompletionStatus', y='Score', palette='Set2')
        plt.title('Score Distribution by Completion Status')
        plt.xlabel('Completion Status')
        plt.ylabel('Score')
        plt.show()
    else:
        print("Score column not found.")


def get_learning_path(student_name, df, label_encoder_student, label_encoder_course):
    """Get learning path for a student."""
    try:
        student_id = label_encoder_student.transform([student_name])[0]
    except ValueError:
        print(f"Student '{student_name}' not found in encoder.")
        return

    student_data = df[df['StudentName_encoded'] == student_id]
    if not student_data.empty:
        completed_courses = label_encoder_course.inverse_transform(
            student_data[student_data['CompletionStatus'] == 'Completed']['Course_encoded']
        ).tolist()
        in_progress_courses = label_encoder_course.inverse_transform(
            student_data[student_data['CompletionStatus'] == 'In Progress']['Course_encoded']
        ).tolist()
        not_started_courses = label_encoder_course.inverse_transform(
            student_data[student_data['CompletionStatus'] == 'Not Started']['Course_encoded']
        ).tolist()

        print(f"\nLearning path for student '{student_name}':")
        print(f"Completed courses: {completed_courses}")
        print(f"Courses in progress: {in_progress_courses}")
        print(f"Courses not started: {not_started_courses}")
    else:
        print(f"No data found for student '{student_name}'")

def get_student_insights(student_name, df, label_encoder_student, label_encoder_course):
    """Provide insights into the student’s weaknesses and strengths."""
    try:
        student_id = label_encoder_student.transform([student_name])[0]
    except ValueError:
        print(f"Student '{student_name}' not found in encoder.")
        return

    student_data = df[df['StudentName_encoded'] == student_id]
    if not student_data.empty:
        strengths = label_encoder_course.inverse_transform(
            student_data[student_data['CompletionStatus'] == 'Completed']['Course_encoded']
        ).tolist()
        weaknesses = label_encoder_course.inverse_transform(
            student_data[student_data['CompletionStatus'] == 'Not Started']['Course_encoded']
        ).tolist()

        print(f"\nInsights for student '{student_name}':")
        print(f"Strengths (completed courses): {strengths}")
        print(f"Weaknesses (not started courses): {weaknesses}")
    else:
        print(f"No data found for student '{student_name}'")

def recommend_courses(student_name, data, label_encoder_student, label_encoder_course):
    """Recommend courses to a student based on their enrollments."""
    try:
        student_id = label_encoder_student.transform([student_name])[0]
    except ValueError:
        print(f"Student '{student_name}' not found in encoder.")
        return []

    student_courses = data[data['StudentName_encoded'] == student_id]['Course_encoded']
    recommended_courses = data[~data['Course_encoded'].isin(student_courses)]['Course_encoded'].unique()
    try:
        recommended_courses = label_encoder_course.inverse_transform(recommended_courses)
      
        recommended_courses = recommended_courses.tolist()
    except Exception as e:
        print(f"Error in label encoding recommended courses: {e}")
        recommended_courses = []
    return recommended_courses

def track_progress(student_name, data, label_encoder_student, label_encoder_course):
    """Track and display a student's progress in enrolled courses."""
    try:
        student_id = label_encoder_student.transform([student_name])[0]
    except ValueError:
        print(f"Student '{student_name}' not found in encoder.")
        return pd.DataFrame()

    student_data = data[data['StudentName_encoded'] == student_id].copy()

   
    try:
        student_data['Course'] = label_encoder_course.inverse_transform(student_data['Course_encoded']).astype(str)
    except Exception as e:
        print(f"Error in inverse transforming 'Course': {e}")
        student_data['Course'] = student_data['Course_encoded'].astype(str)
    
    student_data['CompletionStatus'] = student_data['CompletionStatus'].map({
        'Completed': 'Completed',
        'In Progress': 'In Progress',
        'Not Started': 'Not Started'
    }).astype(str)
    return student_data[['Course', 'CompletionStatus']]


if __name__ == "__main__":
    configure_genai_api()

    file_path = 'dynamic_course_data.csv'
    df = analyze_course_data(file_path)

    if df is not None:
     
        label_encoder_student = LabelEncoder()
        label_encoder_course = LabelEncoder()
        df['StudentName_encoded'] = label_encoder_student.fit_transform(df['StudentName'])
        df['Course_encoded'] = label_encoder_course.fit_transform(df['Course'])

        visualize_course_data(df)

       
        students = ['Dharani', 'Swetha']

        for student in students:
            recommend_courses_list = recommend_courses(student, df, label_encoder_student, label_encoder_course)
            print(f"\nRecommended Courses for {student}: {recommend_courses_list}")

            student_progress = track_progress(student, df, label_encoder_student, label_encoder_course)
            print(f"Progress for {student}:\n{student_progress}")

            get_learning_path(student, df, label_encoder_student, label_encoder_course)
            get_student_insights(student, df, label_encoder_student, label_encoder_course)

       
        courses = df['Course'].unique()
        for course in courses:
            try:
            
                generate_course_quiz(df, course)
            except Exception as e:
                print(f"Error generating quiz for course '{course}': {e}")