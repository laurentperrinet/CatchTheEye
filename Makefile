default: run

run:
	python3 gaze.py

pull_results:
	scp -r laurent@10.164.7.21:Documents/CatchTheEye/_tmp_scanning .

push_dataset:
	rsync -av --delete dataset laurent@10.164.7.21:Documents/CatchTheEye
	rsync -av --delete dataset_faces laurent@10.164.7.21:Documents/CatchTheEye

gitrun: clean
	echo "Running 'make gitrun'" > output.txt
	git pull
	echo `date` >> output.txt
	python3 gaze.py >> output.txt
	echo "Finished running 'make gitrun'" >> output.txt
	echo `date` >> output.txt
	git commit -m ' running without notebook : run parameter scan ' -a ; git push

clean:
	rm -fr _tmp_scanning
