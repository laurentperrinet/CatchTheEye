default: run

run:
	python3 gaze.py

push_dataset:
	scp -r dataset* laurent@10.164.7.21:Documents/CatchTheEye

gitrun:
	python3 gaze.py > output.txt ; git commit -m ' running without notebook : run parameter scan ' -a ; git push

clean:
	rm _tmp_scanning_*
