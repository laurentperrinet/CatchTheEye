default: run

run:
	python3 Regard.py > output.txt ; git commit -m ' running without notebook : run parameter scan ' -a ; git push
