
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import keyboard
import pandas as pd
import xlsxwriter
from datetime import datetime
import os

# Define the path to the YOLO model and Excel file
MODEL_PATH = 'C:/Users/ASUS/Downloads/hhhhh/best (18).pt'
EXCEL_FILE_PATH = "C:/Users/ASUS/Downloads/hhhhh/save.xlsx"  

# Initialize the YOLO model
model = YOLO(MODEL_PATH)

def load_existing_data(filename):
    """ Load existing Excel file to DataFrame if it exists """
    if os.path.exists(filename):
        return pd.read_excel(filename)
    else:
        return pd.DataFrame()

def save_predictions_to_excel(predictions, total_objects, filename=EXCEL_FILE_PATH):
    """ Saves the predictions to an Excel file using xlsxwriter, and handles data formatting correctly. """
    existing_df = load_existing_data(filename)

    data = {
        "Class Name": [],
        "Class ID": [],
        "Time of Prediction": [],
        "Disinfection Time": [],
        "Warning": []
    }

    # Compute the maximum disinfection time for comparison
    max_disinfection_time = max([3 + int(box.cls) if 0 <= int(box.cls) <= 6 else 10 for box in predictions])
    classes_with_max_time = [model.names[int(box.cls)] for box in predictions if (3 + int(box.cls) if 0 <= int(box.cls) <= 6 else 10) == max_disinfection_time]

    for box in predictions:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        disinfection_minutes = 3 + class_id if 0 <= class_id <= 6 else 10
        disinfection_time_str = f"{disinfection_minutes} minutes for disinfection"

        data["Class Name"].append(class_name)
        data["Class ID"].append(class_id)
        data["Time of Prediction"].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        data["Disinfection Time"].append(disinfection_time_str)

        # Set warning if this class has the maximum disinfection time
        if disinfection_minutes == max_disinfection_time:
            warning = f"Maximum disinfection time required for {', '.join(set(classes_with_max_time))} , get it out!"
        else:
            warning = ""
        data["Warning"].append(warning)

    new_df = pd.DataFrame(data)
    separator = pd.DataFrame([["Image", None, None, None, None, total_objects]], columns=list(new_df.columns) + ["Total Objects"])

    if not existing_df.empty:
        df = pd.concat([existing_df, separator, new_df], ignore_index=True)
    else:
        df = pd.concat([separator, new_df], ignore_index=True)

    workbook = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet('Predictions')

    bold = workbook.add_format({'bold': True})  # Define bold format

    headers = list(new_df.columns) + ["Total Objects"]
    for col_num, header in enumerate(headers):
        worksheet.write(0, col_num, header, bold)  # Apply bold format to headers

    for row_num, row_data in enumerate(df.values):
        format = None
        if row_data[0] == "Image":  # Check if the row is the separator
            format = bold  # Use the bold format for the separator row
        for col_num, cell_data in enumerate(row_data):
            if cell_data is not None:
                worksheet.write(row_num + 1, col_num, cell_data, format)  # Apply format conditionally

    workbook.close()

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be accessed.")
        return

    print("Press 'o' to capture and analyze an image. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow('Live Video Feed', frame)

            if keyboard.is_pressed('o'):
                img_resized = cv2.resize(frame, (640, 640))
                results = model.predict(img_resized, conf=0.5)
                total_objects = len(results[0].boxes)

                annotator = Annotator(img_resized, line_width=2, font_size=10)
                for box in results[0].boxes:
                    annotator.box_label(box.xyxy[0], label=model.names[int(box.cls)], color=(0, 255, 0))

                annotated_img = annotator.result()
                cv2.imshow('YOLO V8 Detection', annotated_img)

                if results[0].boxes:
                    save_predictions_to_excel(results[0].boxes, total_objects)

                cv2.waitKey(0)  # Wait for any key press to continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
