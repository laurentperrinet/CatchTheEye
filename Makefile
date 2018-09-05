default: run

run:
	python3 gaze.py

gitrun:
	python3 gaze.py > output.txt ; git commit -m ' running without notebook : run parameter scan ' -a ; git push

clean:
	rm _tmp_scanning_*
