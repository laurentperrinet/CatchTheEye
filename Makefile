default: run

run:
	python3 gaze.py

pull_results:
	scp -r laurent@10.164.7.21:Documents/CatchTheEye/_tmp_scanning .

push_dataset:
	scp -r dataset* laurent@10.164.7.21:Documents/CatchTheEye

gitrun: clean
	echo "Running 'make gitrun'" > output.txt
	git pull
	echo `date` >> output.txt
	python3 gaze.py >> output.txt
	git commit -m ' running without notebook : run parameter scan ' -a ; git push

clean:
	rm -fr _tmp_scanning
