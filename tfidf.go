package tfidf

import (
	"crypto/md5"
	"encoding/hex"
	"math"
	"strings"
)

type Tokenizer interface {
	Seg(text string) []string
	Free()
}

type StdTokenizer struct {
}

// TFIDF tfidf model
type TFIDF struct {
	docIndex  map[string]int         // train document index in TermFreqs
	termFreqs []map[string]int       // term frequency for each train document
	termDocs  map[string]int         // documents number for each term in train data
	n         int                    // number of documents in train data
	stopWords map[string]interface{} // words to be filtered
	tokenizer Tokenizer              // tokenizer, space is used as default
}

func (s *StdTokenizer) Seg(text string) []string {
	return strings.Fields(text)
}

func (s *StdTokenizer) Free() {

}

// New new model with default
func New() *TFIDF {
	return &TFIDF{
		docIndex:  make(map[string]int),
		termFreqs: make([]map[string]int, 0),
		termDocs:  make(map[string]int),
		n:         0,
		tokenizer: &StdTokenizer{},
	}
}

// SetTokenizer sets custom tokenizer
func (f *TFIDF) SetTokenizer(tokenizer Tokenizer) {
	f.tokenizer = tokenizer
}

// AddStopWords add stop words to be filtered
func (f *TFIDF) AddStopWords(words ...string) {
	if f.stopWords == nil {
		f.stopWords = make(map[string]interface{})
	}

	for _, word := range words {
		f.stopWords[word] = nil
	}
}

// AddDocs add train documents
func (f *TFIDF) AddDocs(docs ...string) {
	for _, doc := range docs {
		f.AddDoc(doc)
	}
}

// AddDoc add train document
func (f *TFIDF) AddDoc(doc string) {
	h := hash(doc)
	if f.docHashPos(h) >= 0 {
		return
	}

	termFreq := f.termFreq(doc)
	if len(termFreq) == 0 {
		return
	}

	f.docIndex[h] = f.n
	f.n++

	f.termFreqs = append(f.termFreqs, termFreq)

	for term := range termFreq {
		f.termDocs[term]++
	}
}

// Cal calculate tf-idf weight for specified document
func (f *TFIDF) Cal(doc string) (weight map[string]float64) {
	weight = make(map[string]float64)

	var termFreq map[string]int

	docPos := f.docPos(doc)
	if docPos < 0 {
		termFreq = f.termFreq(doc)
	} else {
		termFreq = f.termFreqs[docPos]
	}

	docTerms := 0
	for _, freq := range termFreq {
		docTerms += freq
	}
	for term, freq := range termFreq {
		weight[term] = tfidf(freq, docTerms, f.termDocs[term], f.n)
	}

	return weight
}

func (f *TFIDF) termFreq(doc string) (m map[string]int) {
	m = make(map[string]int)

	tokens := f.tokenizer.Seg(doc)
	if len(tokens) == 0 {
		return
	}

	for _, term := range tokens {
		if _, ok := f.stopWords[term]; ok {
			continue
		}

		m[term]++
	}

	return
}

func (f *TFIDF) docHashPos(hash string) int {
	if pos, ok := f.docIndex[hash]; ok {
		return pos
	}

	return -1
}

func (f *TFIDF) docPos(doc string) int {
	return f.docHashPos(hash(doc))
}

func hash(text string) string {
	h := md5.New()
	h.Write([]byte(text))
	return hex.EncodeToString(h.Sum(nil))
}

func tfidf(termFreq, docTerms, termDocs, N int) float64 {
	tf := float64(termFreq) / float64(docTerms)
	idf := math.Log(float64(1+N) / (1 + float64(termDocs)))
	return tf * idf
}
