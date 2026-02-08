#ifndef INDEXED_HEAP_H
#define INDEXED_HEAP_H

#include <vector>
#include <algorithm>
#include <iostream>
#include "node.h"

class IndexedHeap {
private:
    std::vector<Node*> data;
    NodeComparator compare;
    
    int parent(int i) {
        return (i - 1) / 2;
    }

    int left(int i) {
        return 2 * i + 1;
    }

    int right(int i) {
        return 2 * i + 2;
    }

    void swap(int i, int j) {
        if(i == j) return;
        
        std::swap(data[i], data[j]);
        data[i]->heap_index = i;
        data[j]->heap_index = j;
    }

    void siftUp(int i) {
        while(i > 0 && compare(data[parent(i)], data[i])) {
            swap(i, parent(i));
            i = parent(i);
        }
    }

    void siftDown(int i) {
        int minIndex = i;
        int l = left(i);
        int r = right(i);
        int n = data.size();

        if(l < n && compare(data[minIndex], data[l])) {
            minIndex = l;
        }
        
        if(r < n && compare(data[minIndex], data[r])) {
            minIndex = r;
        }

        if(i != minIndex) {
            swap(i, minIndex);
            siftDown(minIndex);
        }
    }

public:
    IndexedHeap() {}

    bool empty() const {
        return data.empty();
    }

    size_t size() const {
        return data.size();
    }

    bool contains(Node* v) const {
        return v->heap_index != -1;
    }

    Node* top() const {
        if(data.empty()) return nullptr;
        return data[0];
    }

    void insert(Node* v) {
        if(contains(v)) {
            update(v);
            return;
        }
        
        v->heap_index = data.size();
        data.push_back(v);
        siftUp(v->heap_index);
    }

    Node* pop() {
        if(data.empty()) return nullptr;

        Node* minNode = data[0];
        minNode->heap_index = -1;
        Node* lastNode = data.back();
        data.pop_back();

        if(!data.empty()) {
            data[0] = lastNode;
            data[0]->heap_index = 0;
            siftDown(0);
        }

        return minNode;
    }

    void remove(Node* v) {
        if(!contains(v)) return;

        int i = v->heap_index;
        v->heap_index = -1;

        if(i == data.size() - 1) {
            data.pop_back();
            return;
        }

        Node* lastNode = data.back();
        data.pop_back();
        
        data[i] = lastNode;
        data[i]->heap_index = i;
        siftUp(i);
        siftDown(i);
    }

    void update(Node* v) {
        if(!contains(v)) return;
        siftUp(v->heap_index);
        siftDown(v->heap_index);
    }
};

#endif