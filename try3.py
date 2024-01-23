from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import time
import Pyro5.api



@Pyro5.api.expose
class ImgTrack:
    def output(self):
        global res
        self.visualize_objects_with_axes_and_numbers_interval()
        return res

    def visualize_objects_with_axes_and_numbers_interval(self):
        # Capture a frame from the webcam
        global cap, res
        if cap.isOpened():
            ret, frame = cap.read()

            # Check if the frame was captured successfully
            if not ret:
                print("Error: Could not read frame from webcam.")
                return

            # Preprocess the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)

            edged = cv2.Canny(blur, 40, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            # Find contours
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # Remove contours which are not large enough
            cnts = [x for x in cnts if cv2.contourArea(x) > 100]

            # Result dictionary
            results = {}

            # Draw contours and visualize the center of each rectangle
            for i, cnt in enumerate(cnts):
                box = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)

                # Calculate the center coordinates
                mid_point = np.mean(box, axis=0, dtype=int)

                # Calculate size
                wid = euclidean(box[0], box[1])
                ht = euclidean(box[1], box[2])

                # Visualize the center of the rectangle
                center = (mid_point[0], mid_point[1])
                cv2.circle(frame, center, 3, (0, 255, 0), -1)

                # Draw contours on the frame
                cv2.drawContours(frame, [box.astype("int")], -1, (0, 0, 255), 2)

                # Add object information to the results dictionary
                adjust_factor = 26.5 / 14.0

                results[f"object{i + 1}"] = {
                    "center_location": {"x": int(mid_point[0]) * adjust_factor, "y": int(mid_point[1]) * adjust_factor},
                    "size": {"width": int(wid) * adjust_factor, "height": int(ht) * adjust_factor}
                }

            #global res 
            #res = results
            # Draw x and y axes
            
            res = classify_objects(results)
        
            height, width, _ = frame.shape
            cv2.line(frame, (0, 0), (width, 0), (255, 255, 255), 1)  # x-axis
            cv2.line(frame, (0, 0), (0, height), (255, 255, 255), 1)  # y-axis

            # Add numerical values to the axes
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            x_axis_label_pos = (width - 20, 20)
            y_axis_label_pos = (20, height - 20)

            # Draw x-axis label
            cv2.putText(frame, 'X', x_axis_label_pos, font, font_scale, (255, 255, 255), font_thickness)

            # Draw y-axis label
            cv2.putText(frame, 'Y', y_axis_label_pos, font, font_scale, (255, 255, 255), font_thickness)

            # Draw numerical values on the axes
            for i in range(0, width, 50):
                cv2.putText(frame, str(i), (i, 20), font, font_scale, (255, 255, 255), font_thickness)
            for i in range(0, height, 50):
                cv2.putText(frame, str(i), (20, i), font, font_scale, (255, 255, 255), font_thickness)

            print("showing")
            # Display the frame with contours, center points, axes, and numerical values
            cv2.imshow("Objects with Axes and Numbers", frame)

            # Ensure the OpenCV window is updated
            cv2.waitKey(1)

            # Release the webcam

            # Wait for the specified interval
            #time.sleep(interval_seconds)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return


        return results

def classify_objects(results):
    # Result dictionary for classification
    classified_results = {"Drone": [], "Rocks": []}

    # Loop through all objects in the "results" dictionary
    for key, obj_info in results.items():
        # Extract width and height from the object information
        width = obj_info["size"]["width"]
        height = obj_info["size"]["height"]

        # Check if width is three times larger than the width of any other object (subjected to changes)
        is_large_object = all(
            width > 3 * other_obj["size"]["width"]
            for other_obj_key, other_obj in results.items()
            if other_obj_key != key  # Exclude the current object from comparison
        )

        # Classify objects
        if is_large_object:
            # Append the large object and its properties to the "large_object" list
            classified_results["Drone"].append({
                "center_location": obj_info["center_location"],
                "size": {"width": width, "height": height}
            })
        else:
            # Append the small object and its properties to the "small_objects" list
            classified_results["Rocks"].append({
                "center_location": obj_info["center_location"],
                "size": {"width": width, "height": height}
            })
    print(classified_results)

    return classified_results

# Run the analysis at regular intervals
if __name__ == "__main__":
    global res
    cap = cv2.VideoCapture(0)

    daemon = Pyro5.api.Daemon(host="172.20.10.7")             # make a Pyro daemon
    uri = daemon.register(ImgTrack)         # register Img Track as a Pyro object

    print("Ready. Object uri =", uri)       # print the uri so we can use it in the client later
    daemon.requestLoop()                    # start the event loop of the server to wait for calls


    cap.release()

    # Close the OpenCV windows
    cv2.destroyAllWindows()


