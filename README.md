# TFIDF

## Introduction

This library for NLP allows to calculate Term-Frequency-Inverse Document Frequency, mainly intended for providing an effective way to "weight" a word taking into account its context.

## Usage

```
go get github.com/albertodicagno/tfidf
```
### Sample code

```
package main

import (
	"fmt"

	"github.com/albertodicagno/tfidf"
)

func main() {
	f := tfidf.New()
	f.AddDocs("how are you", "are you fine", "how old are you", "are you ok", "i am ok", "i am file")

	t1 := "you"
	w1 := f.Cal(t1)
	fmt.Printf("weight of %s is %+v.\n", t1, w1)
}

```
## Credits
This is a lightweight library derived from the - apparently abandoned - project  https://github.com/wilcosheh/tfidf