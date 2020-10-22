default: run

IP=10.164.6.114
USER=chloe

run:
	python3 gaze.py

pull_results:
	scp -r ${USER}@${IP}:Documents/CatchTheEye/_tmp_scanning .

push_dataset:
	rsync -av --delete dataset ${USER}@${IP}:Documents/CatchTheEye

pull_dataset:
	rsync -av --delete ${USER}@${IP}:Documents/CatchTheEye/dataset .

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
