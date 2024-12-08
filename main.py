import cv2
from faker import Faker
from sqlalchemy import create_engine, Column, Integer, String, Date, Time, Interval, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random
import threading
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Position
import easyocr

engine = create_engine('sqlite:///Park_Sense_management.db', echo=True)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()
fake = Faker()

# Define the User table
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone_number = Column(String)

    # Relationship to ParkingTicket
    tickets = relationship("ParkingTicket", back_populates="user")

# Define the ParkingTicket table
class ParkingTicket(Base):
    __tablename__ = 'parking_tickets'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    plate_number = Column(String, nullable=False)
    parking_lot = Column(String)
    date_of_entry = Column(Date)
    time_of_entry = Column(Time)
    date_of_exit = Column(Date)
    time_of_exit = Column(Time)
    total_time = Column(Interval)

    # Relationships
    user = relationship("User", back_populates="tickets")
    fines = relationship("Fine", back_populates="ticket")

# Define the Fine table
class Fine(Base):
    __tablename__ = 'fines'

    id = Column(Integer, primary_key=True)
    ticket_id = Column(Integer, ForeignKey('parking_tickets.id'), nullable=False)
    amount = Column(Float, nullable=False)
    issue_date = Column(Date, nullable=False)
    description = Column(String)

    # Relationship to ParkingTicket
    ticket = relationship("ParkingTicket", back_populates="fines")

# Create the tables in the database
Base.metadata.create_all(engine)

# Set the filename for the database.
DATA_FILE = "data.csv"

# This opens the data.csv file if one exits, else it creates one
try:
    info_df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    info_df = pd.DataFrame(columns=[
        'id',
        'plate_number',
        'track_id',
        'parking_lot',
        'date_of_entry',
        'time_of_entry',
        'date_of_exit',
        'time_of_exit',
        'total_time'
    ])

# If the DataFrame exists, reset its contents while keeping the column headers
info_df = info_df.iloc[0:0]

# Initialize at the start of your script (with other global variables)
reader = easyocr.Reader(['en'])  # Initialize once, for English text


def create_line(start_point, end_point):
    line = sv.LineZone(start=start_point, end=end_point,
                       triggering_anchors=(Position.CENTER, Position.CENTER, Position.CENTER, Position.CENTER))
    return line


# This sets the minimum free time allowed in the parking lots. A period less than this will not be recorded in the database.
TOLERANCE = 10
# create useful dictionaries

# For Dictionaries 2-8, track id is the Key and the first time line is crossed in the Value.
dict2 = {}
dict3 = {}
dict4 = {}
dict5 = {}
dict6 = {}
dict7 = {}
dict8 = {}

# The general dictionary in which all LPN's of cars are saved;
dict_general = {}

# This dictionaries saves the state of the object around it, True if the object has crossed otherwise false. Key is track id and Value is true/false.
isCrossedA = {}


# street entrance ( line1, line5 )
def process_tracker_state_line(detections, line_counter, dict_general, labels):
    global reader  # Use the global reader instance

    line_counter.trigger(detections=detections)
    tracker_state_line = line_counter.tracker_state

    for track_id, state in tracker_state_line.items():
        if state:  # object has crossed the line
            if track_id not in dict_general:
                # Get the bounding box for this track_id
                for xyxy, _, _, _, tracker_id, _ in detections:
                    if tracker_id == track_id:
                        # Extract the region containing the vehicle
                        x1, y1, x2, y2 = map(int, xyxy)
                        vehicle_roi = frame[y1:y2, x1:x2]

                        try:
                            # Attempt to read license plate from the ROI
                            results = reader.readtext(vehicle_roi)
                            if results:
                                # Take the most confident result
                                plate_text = max(results, key=lambda x: x[2])[1]
                                # Clean up the text (remove spaces, special characters)
                                plate_text = ''.join(e for e in plate_text if e.isalnum()).upper()

                                if len(plate_text) >= 4:  # Minimum length for valid plate
                                    dict_general[track_id] = plate_text
                                    print(f"Detected plate: {plate_text}")
                                else:
                                    # Fallback to temporary ID if plate detection fails
                                    temp_id = f"TEMP_{track_id}"
                                    dict_general[track_id] = temp_id
                            else:
                                temp_id = f"TEMP_{track_id}"
                                dict_general[track_id] = temp_id
                        except Exception as e:
                            print(f"Error reading plate: {e}")
                            temp_id = f"TEMP_{track_id}"
                            dict_general[track_id] = temp_id

            # Update labels
            if str(track_id) in labels:
                index = labels.index(str(track_id))
                plate_number = dict_general[int(track_id)]
                labels[index] = plate_number


def frame_change_start(detections, line_counter, dict_line):
    # reference to the relevant global variables
    global isCrossedB1, info_df

    # activate the in-built line trigger function which takes detections as a param
    line_counter.trigger(detections=detections)

    # track the state of objects with respect to the line. True if objects are to the left of the line otherwise false. Oriented with start point on top.
    tracker_state_line = line_counter.tracker_state

    # iterate through all the objects around the line and record the first time the detected object crosses the line.
    for track_id, state in tracker_state_line.items():
        if state:
            if track_id not in dict_line:
                time_out = datetime.now().strftime("%H:%M:%S")
                dict_line[track_id] = time_out


def transfer_plate_number(detections, line_counter, dict_first, dict_second, dict_general, labels):
    # activate the in-built line trigger function which takes detections as a param
    line_counter.trigger(detections=detections)

    # Track the state of objects with respect to the line. True if objects are to the left of the line otherwise false. Oriented with start point on top.
    tracker_state_line = line_counter.tracker_state

    # iterate through all the objects around the line and record the first time the detected object crosses the line.
    for track_id, state in tracker_state_line.items():
        if state:
            if track_id not in dict_second:
                dict_second[track_id] = datetime.now().strftime("%H:%M:%S")

            # Tolerance for time comparison (+/- 20 seconds) between the exit line in the previous frame and the entry line in the present frame.
            tolerance = timedelta(seconds=10)

            # Transfer the LPN of a car that has a time difference of less that the set tolerance.
            for key1, time1 in dict_first.items():
                for key2, time2 in dict_second.items():
                    delta = abs(time_to_timedelta(time1) - time_to_timedelta(time2))
                    # If the time difference is within the limit, add the track id(detected when the car entered the new frame) and plate number(from the previous frame's car) in the general dictionary
                    if delta <= tolerance:
                        platenum = dict_general[key1]
                        dict_general[key2] = platenum

            # update the label with the LPN if the track_id is in the general dictionary.
            for _, _, _, _, tracker_id, _ in detections:
                if tracker_id in dict_general.keys() and str(tracker_id) in labels:
                    index_to_replace = labels.index(str(tracker_id))
                    plate_num = dict_general[int(tracker_id)]

                    labels[index_to_replace] = plate_num


def update_exit_info(plate_num, time_out):
    global info_df, TOLERANCE

    # select the current object from the database and update its time out.
    info_df.loc[(info_df['plate_number'] == plate_num) & (info_df['time_of_exit'] == 0), 'time_of_exit'] = time_out
    # Set the entry time and date
    time_in_str = info_df.loc[info_df['plate_number'] == plate_num, 'time_of_entry'].iloc[
        -1]  # Assuming there's only one value
    date_of_entry = info_df.loc[info_df['plate_number'] == plate_num, 'date_of_entry'].iloc[-1]

    # Set the exit date
    date_of_exit = datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(' ')[0]
    info_df.loc[info_df['plate_number'] == plate_num, 'date_of_exit'] = date_of_exit

    entry_datetime = datetime.strptime(date_of_entry + " " + time_in_str, "%Y-%m-%d %H:%M:%S")
    exit_datetime = datetime.strptime(date_of_exit + " " + time_out, "%Y-%m-%d %H:%M:%S")
    time_difference = exit_datetime - entry_datetime

    if time_difference.total_seconds() < TOLERANCE:
        # Delete the specific row where the time difference is less than TOLERANCE
        info_df = info_df.drop(info_df[(info_df['plate_number'] == plate_num) & (info_df['total_time'] == 0)].index)
    else:
        # else update the database with the total time spent.
        info_df.loc[
            (info_df['plate_number'] == plate_num) & (info_df['total_time'] == 0), 'total_time'] = time_difference


def create_record(plate_num, track_id, parking_lot):
    global info_df

    # Add detection confidence if it's a temporary ID
    is_temp = plate_num.startswith("TEMP_")

    data_dict = {
        'id': len(info_df) + 1,
        'plate_number': plate_num,
        'track_id': track_id,
        'parking_lot': parking_lot,
        'is_temporary': is_temp,  # Flag for temporary IDs
        'date_of_entry': datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(' ')[0],
        'time_of_entry': datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(' ')[1],
        'date_of_exit': 0,
        'time_of_exit': 0,
        'total_time': 0
    }
    return data_dict


def convert_to_timedelta(time_str):
    """Convert a time string 'HH:MM:SS' to a timedelta object."""
    if time_str == "0":
        return timedelta(0)
    time_parts = list(map(int, time_str.split(":")))
    if len(time_parts) == 2:  # Handle cases where the time is in minutes:seconds format
        return timedelta(minutes=time_parts[0], seconds=time_parts[1])
    return timedelta(hours=time_parts[0], minutes=time_parts[1], seconds=time_parts[2])
def update_csv(data_dict):
    """
    Updates the database with a newly created record.

    :param dict data_dict: The dictionary containing a new record;
    :return: None
    :rtype: None
    """
    global info_df
    new_data = pd.DataFrame(data_dict, index=[0])  # Convert dictionary to DataFrame
    info_df = pd.concat([info_df, new_data], ignore_index=True)

    with open(DATA_FILE, "a") as file:
        info_df.to_csv(file, index=False)

    if data_dict['time_of_entry']:
        # Convert total_time string to timedelta
        time_of_entry = convert_to_timedelta(data_dict['time_of_entry'])

        # Check if total_time exceeds 1 hour (3600 seconds) and add fine if needed
        if total_time.total_seconds() > 30:  # Check if time exceeds 1 hour
            add_ticket_to_db(data_dict)
        else:
            print('No fine is applicable')
    else:
        print('data_dict["total_time"] does not exist')

def add_ticket_to_db(data_dict):
    """Add a parking ticket, associated user, and fine to the database without conditions."""

    # Check for existing user or create a new one
    user = session.query(User).filter_by(id=data_dict['id']).first()
    if not user:
        user = User(
            id=data_dict['id'],
            name=fake.name(),
            email=fake.email(),
            phone_number=fake.phone_number()
        )
        session.add(user)
        session.commit()

    # Parse dates and times for entry
    date_of_entry = datetime.strptime(data_dict['date_of_entry'], "%Y-%m-%d").date()
    time_of_entry = datetime.strptime(data_dict['time_of_entry'], "%H:%M:%S").time()
    date_of_exit = None if data_dict['date_of_exit'] == 0 else datetime.strptime(data_dict['date_of_exit'], "%Y-%m-%d").date()
    time_of_exit = None if data_dict['time_of_exit'] == 0 else datetime.strptime(data_dict['time_of_exit'], "%H:%M:%S").time()

    # Calculate total time if exit times are provided
    if date_of_exit and time_of_exit:
        entry_datetime = datetime.combine(date_of_entry, time_of_entry)
        exit_datetime = datetime.combine(date_of_exit, time_of_exit)
        total_time = exit_datetime - entry_datetime
    else:
        total_time = None

    # Add parking ticket record
    new_ticket = ParkingTicket(
        user_id=user.id,
        plate_number=data_dict['plate_number'],
        parking_lot=data_dict['parking_lot'],
        date_of_entry=date_of_entry,
        time_of_entry=time_of_entry,
        date_of_exit=date_of_exit,
        time_of_exit=time_of_exit,
        total_time=total_time
    )
    session.add(new_ticket)
    session.commit()

    # Add a fine record unconditionally
    new_fine = Fine(
        ticket_id=new_ticket.id,
        amount=round(random.uniform(50, 200), 2),  # Random fine amount
        issue_date=datetime.now().date(),
        description="Parking fine issued"
    )
    session.add(new_fine)
    session.commit()

    print("Ticket, user, and fine records added to the database unconditionally.")

def parking_lot_true(track_id, parking_name, dict_isCrossed):
    global dict_general, info_df
    plate_num = dict_general[int(track_id)]
    data_dict = create_record(plate_num, track_id, parking_name)
    update_csv(data_dict)
    # changing the isCrossed status to true, the moment the car enteres the parking space
    dict_isCrossed[track_id] = True


def parking_lot(detections, line_counter, dict_isCrossed, parking_name):
    global info_df, dict2, dict3, dict_general

    # activate the in-built line trigger function which takes detections as a param
    line_counter.trigger(detections=detections)

    # Track the state of objects with respect to the line. True if objects are to the Top of the line otherwise false. Oriented with start point on Left to Right.
    tracker_state_line = line_counter.tracker_state

    # iterate through all the objects around the line and record the first time the detected object crosses the line.
    for track_id, state in tracker_state_line.items():

        # If the line is not crossed, make dict_isCrossed as False
        if track_id not in dict_isCrossed:
            dict_isCrossed[track_id] = False

            # If car has crossed the line, check their respective dictionary and call function(parking_lot_true)
        if state:
            if dict_isCrossed[track_id] == False:
                parking_lot_true(track_id, parking_name, dict_isCrossed)

        elif not state and dict_isCrossed[track_id] == True:
            time_out = datetime.now().strftime("%H:%M:%S")
            dict_isCrossed[track_id] = False
            plate_num = dict_general[int(track_id)]
            # Calculate the total duration spent by an object by updating the time out.
            if (info_df['plate_number'] == plate_num).any():
                update_exit_info(plate_num, time_out)

def run_tracker_in_thread1(filename, model):
    global info_df, dict2, dict3, dict4, dict5, dict6, dict7, dict8, dict_general, frame
    global isCrossedA

    # Set the constraints as dynamic
    vid = cv2.VideoCapture(filename)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Place the lines on frames as desired in first frame
    LINE1_START = sv.Point((0.8 / 8) * width, (0.8) * height)
    LINE1_END = sv.Point((4.5 / 8) * width, (0.8) * height)

    LINE8_START = sv.Point((1.5 / 8) * width, (0.6 / 4) * height)
    LINE8_END = sv.Point((4 / 8) * width, (0.6 / 4) * height)

    LINEA_START = sv.Point((4 / 8) * width, (0.5 / 4) * height)
    LINEA_END = sv.Point((4.25 / 8) * width, (2.5 / 4) * height)

    # Create line1
    line1_counter = create_line(start_point=LINE1_START, end_point=LINE1_END)

    line8_counter = create_line(start_point=LINE8_END, end_point=LINE8_START)
    # create lineA
    lineA_counter = create_line(start_point=LINEA_START, end_point=LINEA_END)

    # Annotating the lines to make desirable structure.
    line1_annotator = line8_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0,
                                                             color=Color.BLACK, text_color=Color.WHITE)
    lineA_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0, color=Color.RED)

    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=2, text_position=Position.TOP_CENTER)

    # Drawing the bounding box of detections as a dot.
    dot_annotator = sv.DotAnnotator()

    # Iterating through each frame of the IP camera live stream.
    for result in model.track(source=filename, persist=True, stream=True, tracker="bytetrack.yaml"):

        frame = result.orig_img
        detections = sv.Detections.from_ultralytics(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Set the desired detection classes (2- car,5- bus,7- truck)
        selected_classes = [2, 5, 7]
        detections = detections[np.isin(detections.class_id, selected_classes)]

        line1_annotator.annotate(frame=frame, line_counter=line1_counter)

        line8_annotator.annotate(frame=frame, line_counter=line8_counter)
        lineA_annotator.annotate(frame=frame, line_counter=lineA_counter)

        # Save the tracker id's generated by the detection model
        labels = [
            f"{tracker_id}"
            for _, _, _, _, tracker_id, _ in detections
        ]

        if len(detections) > 0 and result.boxes.id is not None:
            # Create the line1
            process_tracker_state_line(detections, line1_counter, dict_general, labels)
            # Create the lineA
            parking_lot(detections, lineA_counter, isCrossedA, "parking_A")

            # Crete a copy of the frame and annotate it as desired.
        annotated_frame = frame.copy()
        annotated_frame = dot_annotator.annotate(
            scene=frame,
            detections=detections,
        )
        annotated_frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # Save the updated DataFrame to the CSV file after processing all objects in the frame
        with open(DATA_FILE, "w") as file:
            info_df.to_csv(file, index=False)

        # Display the annotated frame
        cv2.imshow(f"Tracking_Stream_{filename}", annotated_frame)

        # Set 'q' to quit the video display window.
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def time_to_timedelta(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def main():
    # Load the models
    model1 = YOLO("yolov8l.pt")

    # Set the IP camera
    cap1 = 'http://192.168.150.86:9090/video'

    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread1, args=(cap1, model1), daemon=True)

    # Start the tracker threads
    tracker_thread1.start()

    # Wait for the tracker threads to finish
    tracker_thread1.join()

    # Clean up and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
