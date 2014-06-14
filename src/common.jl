module Common

    using JSON

    type Ranking
        class::Int64
        value::Float64
    end

    type TopKNode
        prev::TopKNode
        next::TopKNode
        weight::Float64
        item::Any
    end

    type TopKQueue
        head::TopKNode
        tail::TopKNode
        op::Function
        length::Int64
        lenLimit::Int64
        TopKQueue(op::Function, lenLimit::Int64) = new(nothing,nothing,op,0,lenLimit)
    end

    function push(q::TopKQueue, item::Any, weight::Float64)

        if q.length == q.lenLimit

            if !q.op(q.tail.weight, weight)
                return nothing
            else q.op(q.tail.weight, weight)
                t = q.tail
                q.tail = q.tail.prev
                q.tail.next = nothing
                t.prev = nothing
                t.next = nothing
            end
        else
            q.length += 1
        end

        front = q.head
        if q.op(front.weight, weight)
            q.head = TopKNode(nothing, front, item, weight)
            return q
        else
            preFront = front
            while front.next != nothing && !q.op(front.weight, weight)
                preFront = front
                front = front.next
            end
            front.prev = TopKNode(preFront,front,item,weight)
            preFront.next = front.prev
            return front.prev
        end

    end

    function pop(q::TopKQueue)
        if q.length > 0
            q.length -= 1
            h = q.head
            q.head = q.head.next

            if q.head != nothing
                q.head.prev = nothing
            end

            return h.item, h.weight
        else
            return nothing,nothing
        end
    end

    function queueIter(q::TopKQueue)
        h = q.head
        while h != nothing
            produce( (h.item, h.weight,) )
            h = h.next
        end
    end

    function each(q::TopKQueue)
        @task queueIter(q)
    end

export Ranking,TopKQueue,TopKNode, push, pop, each
end
