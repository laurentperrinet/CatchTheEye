default: run

run:
	python3 gaze.py

push_dataset:
	scp -r dataset* laurent@10.164.7.21:Documents/CatchTheEye

gitrun: clean
	echo "Running 'make gitrun'" > output.txt
<<<<<<< HEAD
	git pull
	python3 gaze.py >> output.txt 
	git pull; git commit -m ' running without notebook : run parameter scan ' -a ; git push
=======
	echo `date`
	python3 gaze.py >> output.txt ; git commit -m ' running without notebook : run parameter scan ' -a ; git push
>>>>>>> 9b65115e43907db4db1d65cca497f8818b0f74dd

clean:
	rm _tmp_scanning_*
