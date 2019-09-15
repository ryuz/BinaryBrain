#!/bin/bash

./clean.sh
./copy_src.sh

python3 setup.py build
python3 setup.py sdist

while true;do
    echo -n "upload testPyPI?(yes/no):"
    read answer
    case $answer in
        yes)
            echo "upload testpypi"
            python3 setup.py sdist upload -r testpypi
            break
            ;;
        no)
            echo "don't upload"
            break
            ;;
        *)
            ;;
    esac
done


while true;do
    echo -n "upload PyPI?(yes/no):"
    read answer
    case $answer in
        yes)
            echo "upload pypi"
            python3 setup.py sdist upload -r pypi
            break
            ;;
        no)
            echo "don't upload"
            break
            ;;
        *)
            ;;
    esac
done

