- install required libraries:
pip install -r requirements.txt

- run: main.py
	+ change map and other parameters inside source code
	+ add dynamic obstacles: add into dynamic obstacle list (dynamic_obs_list) the following parameters:
		(x, y) - dynamic obstacle's starting position
		(width, height) - dynamic obstacle's size
		dynamic obstacle's moving direction (1 - up, 2 - left, 3 - right, 4 - down)
		dynamic obstacle's velocity
	+ edit map: 
		left click to change the state of cell (free cell -> static obstacle and vice versa)
		right click for change the robot's starting position
	+ change running speed: left & right arrow key
