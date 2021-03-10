INSTRUCTION=

train-instruction:
	python -m src.train_maze_from_instruction --instruction=$(INSTRUCTION)