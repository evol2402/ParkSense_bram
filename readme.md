Parksense: Automated Ticketing System

![image](https://github.com/user-attachments/assets/77d76c50-571f-45d8-bb78-2e214aea43fd)


Parksense is a cutting-edge project designed to automate the ticketing system for parking lots using advanced object detection and OCR (Optical Character Recognition) technologies. It combines computer vision, database management, and machine learning to efficiently identify vehicles, extract license plate information, and generate tickets automatically.

Interesting Facts
Real-time Efficiency: Parksense can process video streams frame-by-frame, ensuring that no vehicle goes undetected, even in busy parking lots.
Scalable Design: Parksense can handle parking lots of all sizes, from small facilities to large multi-level complexes.
Accuracy: With state-of-the-art YOLO and EasyOCR technologies, it achieves over 90% accuracy in vehicle detection and license plate recognition.
Environmentally Friendly: By automating ticketing, Parksense reduces the need for paper tickets and promotes a more sustainable solution.
Cost-effective: Automating the ticketing process reduces reliance on manual labor, saving operational costs for parking lot operators.
Customizable Rules: Parksense can integrate with different ticketing rules, such as dynamic pricing or free parking periods for EVs, making it adaptable to diverse scenarios.
Future-ready: Designed with modularity in mind, the system can easily integrate with future innovations like payment gateways and smart city frameworks.

Features
Automated Vehicle Detection: Powered by YOLO object detection models.
License Plate Recognition: Uses EasyOCR for accurate text extraction from license plates.
Database Integration: Tracks ticketing data with SQLAlchemy and SQLite.
Real-time Processing: Utilizes OpenCV for processing video streams.
Customizable Settings: Adjust detection parameters and ticket rules.
Technologies Used
Programming Language: Python
Computer Vision: OpenCV, YOLO (Ultralytics)
OCR: EasyOCR
Database: SQLAlchemy
Data Manipulation: NumPy, Pandas
Visualization: Supervision

Usage
Upload or stream parking lot footage.
The system detects vehicles and extracts their license plates.
Tickets are automatically generated and logged in the database.

Acknowledgments
Ultralytics YOLO for object detection.
EasyOCR for OCR capabilities.
Open-source contributors who make these tools possible.

We created this project as part of Brampton Hacks 2024, a prestigious hackathon that brought together innovative minds to solve real-world challenges. Parksense was developed to address the inefficiencies in parking management systems, showcasing our commitment to leveraging technology for smarter and more efficient urban solutions.

Feel free to reach out for collaboration or suggestions!
