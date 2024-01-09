# Visual Perception Problem: 
This project is a robot perception problem in which a robot needs to map and navigate a simulated maze environment using its camera input. The goal of the project is to navigate to a target location; however, the location’s coordinates are not provided. Instead, we are given 4 images taken from different angles at that location. The robot needs to compare the images to what it has previously seen using appropriate robot perception models and algorithms.

Our Solution is to have an odometry based approach to explore the maze. This means that the current position and pose of the robot is saved by the number of forward, backward movement and heading is calculated by the number of left and right turns taken by the robot. We then perform visual place recognition using SIFT and BoVW feature matching on the robot's POV (Every 5 frames) and the target images. Once we have the closest matches, our program navigates to the position of that closest match.

The program can be divided into 2 phases: Exploration and Navigation

## Exploration: 
### Mapping the Game Environment:
Occupancy Grid is a 2D array containing information about the contents of the maze, either visited, unvisited, obstacle, or unknown. We created an interactive Map View. It is a 2D visual representation of the map created by converting the above array into an image using numpy, and drawing with opencv. The program saves once every 15 frames using opencv's imWrite function, with the filename encoding the timestep, x-y coordinates, and orientation. 

### Wall Detection: 
A loop iterates through a sensing width, projecting points in front of the robot in evenly spaced intervals. For each iteration, points in 3D space are created and projected onto the camera plane. Offset calculations are performed based on the robot's current position. A section of the fpv image corresponding to the projected points is extracted. If the image section consists of only floor tiles, the corresponding grid cell is marked as an unvisited location. Otherwise, it is marked as an obstacle. Virtual range sensing lines are also displayed as a visual aid. The wall detection is quite noisy due to the low resolution. OpenCV’s blurring and morphological operations (erode and dilate) are used to remove erroneous wall markers and smooth out the walls.

### Assisted Control:
The robot’s heading and x-y position are calculated internally using trigonometry. Moving forward is also prevented if any of the grid cells in front have been marked as obstacles. The original direction keys were mapped in such a way that it would take 147 left keys to make a complete 360 degree turn. The direction keys are mapped so the robot turns 90 degrees when turning left or right, allowing the user to navigate easier.

### Automated Exploration: 
The AStar algorithm is used for path planning to explore the environment efficiently. The robot explores in a spiral pattern, adjusting the exploration radius based on the success or failure of previous paths. The robot occasionally performs a 360-degree scan to refine the occupancy map and get a good look at the walls.The occupancy map is used for collision avoidance during path planning. If a path fails, the script refines the occupancy map, erodes it, or attempts to move to a nearby location, depending on the number of consecutive failures. A checkpoint system is implemented to store "good" checkpoints during exploration, allowing the robot to return to a previous spot if needed. Exploration is considered complete when a specified percentage of the map is explored or a certain time threshold is reached. After completion, the script transitions to a post-exploration state for further processing. This video demonstrates a fully automated explorer and navigator robot : https://www.youtube.com/watch?v=CdN3MW6t1S0

## Navigation
### Feature Detection and Feature Matching
The Faiss library is defined for performing k-means clustering using the Faiss library. It is used to cluster SIFT descriptors into visual words.
SIFT features are computed for each image, and descriptors are stored. Visual words are built by clustering SIFT descriptors using the Faiss KMeans algorithm. Histograms are calculated for each training image based on the visual words. (Target image also has to run through the algorithm to get descriptors and visual words) Tentative matches are identified by comparing the histogram of the query image with those of training images. A brute force matcher is used to find good matches between descriptors of the query image and those of tentative matches. The image with the highest count of good matches is selected as the best match.

### Automated Navigation: (A*)
Target images are used to query the database of geotagged images using the bag of Visual Words(BoVW) feature matching algorithm. This gives up to 4 possible locations; the coordinates closest to the average of these locations is chosen as the goal location. The layout of the maze is defined by the occupancy map in a grid state of the cells represented as Unexplored, Visited and Obstacles. Maze is solved using the A-Star pathfinding algorithm. Start and end goal coordinates and the occupancy map are input into the function. A cost of 1 is assigned to move from one node to another. (in all directions). Turning costs are introduced to influence the selection of paths with smoother turns. A* search algorithm is executed using a priority queue. The program iterates through cells and updating the cost to reach them, it estimates the total cost using Manhattan distance as Heuristic and stores the reached cells and their costs. The path is a list of coordinates representing the optimal route through the grid. It is returned as an empty if the goal is not reachable. Once the path to the goal is found, the robot automatically follows the path and checks in once arriving at the target.


## Additional Comments:
### How did we implement the cost function so that we can choose a path with fewer turns?
It takes 37 frames to perform a 90° turn, i.e. you have to return Action.LEFT from act() 37 times. That’s a lot compared to just doing Action.FORWARD once to go forward!
In the A* path search algorithm, we added an extra dimension so that each node includes the heading in addition to the x-y coordinate. When adding the neighbor nodes into the frontier set, the cost is 1 if that neighbor is reached by moving in the same direction, or 37+1 if it involves turning left or right.

### Did we have any feedback mechanisms for if the bot collided with the wall?
If the bot collides with a wall in front of it, neither the marker on the map, nor the robot moves. The heading and position also remain unchanged. If you mean what if it gets stuck during the navigation stage, the turning cost function makes sure the robot doesn't get stuck while navigating. 

### Is this virtual line projection for wall detection based on the color (pixel intensities) at the given line locations? Or some other aspects as well?
It’s just based on the color, it literally just checks if the pixel is one of two colors found on the floor ([239, 239, 239] for the white squares and [224, 186, 162] for the blue squares). Admittedly, this isn’t very robust, and only works here because we have a uniform floor and there are no lighting effects or camera noise. In a real-world setting, more advanced logic would be needed.
