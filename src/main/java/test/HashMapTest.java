package test;

import java.util.HashMap;

/**
 * Created by Administrator on 2016/1/17.
 */
public class HashMapTest extends HashMap<String,Integer> {
    public void test(){
        put("leaf", 12);
        put("cheng", 123);
    }

    public static void main(String[] args) {
        HashMapTest test = new HashMapTest();
        test.test();
        System.out.println(test.keySet());

    }

}
