build:
	docker build --network host --compress -t jodafons/pytorch-rxwgan:latest .
	docker build --network host --compress -t jodafons/pytorch-rxwgan:base .
push:
	docker push jodafons/pytorch-rxwgan:latest
	docker push jodafons/pytorch-rxwgan:base
