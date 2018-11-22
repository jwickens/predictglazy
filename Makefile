init:
	. ./env/bin/activate

test:
	py.test --ignore=ignite/ -f