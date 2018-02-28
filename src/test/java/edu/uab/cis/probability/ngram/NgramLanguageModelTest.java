package edu.uab.cis.probability.ngram;

import static com.google.common.collect.Lists.charactersOf;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.Lists;

import edu.uab.cis.probability.ngram.NgramLanguageModel.Representation;
import edu.uab.cis.probability.ngram.NgramLanguageModel.Smoothing;

public class NgramLanguageModelTest {

  @Test(timeout = 10000)
  public void testCharacter2gram() {
    NgramLanguageModel<Character> model =
        new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE);
    model.train(charactersOf("babbaaaa"));
    Assert.assertEquals((5.0 / 8.0) * (3.0 / 5.0) * (3.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0)
        * (1.0 / 3.0), model.probability(charactersOf("aaaabb")), 1e-10);
  }

  @Test(timeout = 10000)
  public void testCharacter2gramLaplace() {
    NgramLanguageModel<Character> model =
        new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.LAPLACE);
    model.train(charactersOf("babbaaaa"));
    Assert.assertEquals((6.0 / 10.0) * (4.0 / 7.0) * (4.0 / 7.0) * (4.0 / 7.0) * (2.0 / 7.0)
        * (2.0 / 5.0), model.probability(charactersOf("aaaabb")), 1e-10);
  }

  @Test(timeout = 10000)
  public void testInteger4gramLogprobLaplace() {
    NgramLanguageModel<Integer> model =
        new NgramLanguageModel<>(4, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
    model.train(Arrays.asList(1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0));
    Assert.assertEquals(
        Math.log((8.0 / 18.0) * (6.0 / 9.0) * (4.0 / 7.0) * (2.0 / 5.0) * (1.0 / 3.0) * (1.0 / 3.0)
            * (1.0 / 3.0)),
        model.probability(Arrays.asList(1, 0, 0, 0, 0, 0, 1)),
        1e-10);
  }

  @Test
  public void testCharacter2gramNoTrain() {
    NgramLanguageModel<Character> model =
        new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE);
    model.train(charactersOf(""));
    Assert.assertEquals(0, model.probability(charactersOf("aaaabb")), 1e-10);
  }
  
  @Test
  public void testCharacter2gramNoTest() {
    NgramLanguageModel<Character> model =
        new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE);
    model.train(charactersOf("aaabb"));
    Assert.assertEquals(0, model.probability(charactersOf("")), 1e-10);
  }
  
/////////////////////////
@Test(timeout = 10000)
public void testCharacter3gramLogprob() {
NgramLanguageModel<Character> model = new NgramLanguageModel<>(3,
Representation.LOG_PROBABILITY,
Smoothing.NONE);
model.train(charactersOf("aaba"));
Assert.assertEquals(Math.log((3.0 / 4.0) * (1.0 / 3.0) * (0.0 / 1.0)),
model.probability(charactersOf("abc")), 1e-10);
}

@Test(timeout = 10000)
public void testCharacter3gramLogprobLaplace() {
NgramLanguageModel<Character> model = new NgramLanguageModel<>(3,
Representation.LOG_PROBABILITY,
Smoothing.LAPLACE);
model.train(charactersOf("aaabaaabaaac"));
Assert.assertEquals(Math.log((10.0 / 15.0) * (7.0 / 12.0) * (3.0 / 9.0)
* (1.0 / 5.0) * (1.0 / 3.0) * (1.0 / 3.0) * (3.0 / 9.0) * (1.0 / 5.0)),
model.probability(charactersOf("aabdaabc")), 1e-10);
}

@Test(timeout = 10000)

public void testCharacter3gramAll() {
NgramLanguageModel<Character> model = new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.NONE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(Double.NaN, model.probability(charactersOf("bcbacba")), 1e-10);

model = new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.NONE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(Double.NaN, model.probability(charactersOf("bcbacba")), 1e-10);

model = new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.LAPLACE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(((5.0 / 15.0) * (3.0 / 7.0) * (1.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0) * (1.0 / 3.0) * (3.0 / 5.0)),
model.probability(charactersOf("bcbacba")), 1e-10);

model = new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(Math.log((5.0 / 15.0) * (3.0 / 7.0) * (1.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0) * (1.0 / 3.0) * (3.0 / 5.0)),
model.probability(charactersOf("bcbacba")), 1e-10);


}

//////

@Test(timeout = 10000)
public void testCharacter3gramAbsence(){
NgramLanguageModel<Character> model0 =
new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.NONE);
model0.train(charactersOf("abbabababab"));
Assert.assertEquals(0.0, model0.probability(charactersOf("abbababababc")), 1e-10);


model0 = new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.NONE);
model0.train(charactersOf("abbabababab"));
Assert.assertEquals(Double.NEGATIVE_INFINITY, model0.probability(charactersOf("abbababababc")), 1e-10);

NgramLanguageModel<Character> model1 =
new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.NONE);
model1.train(charactersOf("ab"));
Assert.assertEquals(Double.NaN, model1.probability(charactersOf("abcde")), 1e-10);

NgramLanguageModel<Character> model2 =
new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.NONE);
model2.train(charactersOf("ab"));  
Assert.assertEquals(Double.NaN , model2.probability(charactersOf("abcde")), 1e-10);
}

@Test(timeout = 10000)
public void testCharacter3gramAbsenceWithLabplace(){
NgramLanguageModel<Character> model1 =
new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.LAPLACE);
model1.train(charactersOf("ab"));
Assert.assertEquals((2.0 / 4.0) * (2.0 / 3.0) * (1.0 / 3.0) * (1.0 / 2.0) * (1.0 / 2.0), model1.probability(charactersOf("abcde")), 1e-10);

NgramLanguageModel<Character> model2 =
new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model2.train(charactersOf("ab"));  
Assert.assertEquals(Math.log((2.0 / 4.0) * (2.0 / 3.0) * (1.0 / 3.0) * (1.0 / 2.0) * (1.0 / 2.0)) , model2.probability(charactersOf("abcde")), 1e-10);
}

@Test(timeout = 10000)
public void testWithSameTrainandTest(){
NgramLanguageModel<Character> model1 =
new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE);
model1.train(charactersOf("aab"));
Assert.assertEquals((2.0 / 3.0) * (1.0 / 2.0) * (1.0 / 2.0), model1.probability(charactersOf("aab")), 1e-10);

NgramLanguageModel<Character> model2 =
new NgramLanguageModel<>(2, Representation.LOG_PROBABILITY, Smoothing.NONE);
model2.train(charactersOf("aab"));
Assert.assertEquals(Math.log((2.0 / 3.0) * (1.0 / 2.0) * (1.0 / 2.0)), model2.probability(charactersOf("aab")), 1e-10);     

NgramLanguageModel<Character> model3 =
new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.LAPLACE);
model3.train(charactersOf("aab"));
Assert.assertEquals((3.0 / 5.0) * (2.0 / 4.0) * (2.0 / 4.0), model3.probability(charactersOf("aab")), 1e-10);   

NgramLanguageModel<Character> model4 =
new NgramLanguageModel<>(2, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model4.train(charactersOf("aab"));
Assert.assertEquals(Math.log((3.0 / 5.0) * (2.0 / 4.0) * (2.0 / 4.0)), model4.probability(charactersOf("aab")), 1e-10);     
}

@Test(timeout = 10000)
public void testString1gram(){
NgramLanguageModel<String> model1 =
new NgramLanguageModel<>(1, Representation.PROBABILITY, Smoothing.NONE);
model1.train(Arrays.asList("this", "is", "a", "test"));	    
Assert.assertEquals((1.0 / 4.0) * (1.0 / 4.0) * (1.0 / 4.0)* (1.0 / 4.0), model1.probability(Arrays.asList("is", "this", "test", "a")), 1e-10);

NgramLanguageModel<String> model2 =
new NgramLanguageModel<>(1, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model2.train(Arrays.asList("this", "is", "a", "test"));	    
Assert.assertEquals(Math.log((2.0 / 8.0) * (2.0 / 8.0) * (2.0 / 8.0)* (2.0 / 8.0)), model2.probability(Arrays.asList("is", "this", "test", "a")), 1e-10);   

NgramLanguageModel<String> model3 =
new NgramLanguageModel<>(1, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model3.train(Arrays.asList("this", "is", "a", "test"));	    
Assert.assertEquals(Math.log((2.0 / 8.0) * (1.0 / 8.0) * (1.0 / 8.0)* (1.0 / 8.0)), model3.probability(Arrays.asList("is", "asd", "dwew", "5656")), 1e-10); 

NgramLanguageModel<String> model4 =
new NgramLanguageModel<>(1, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model4.train(Arrays.asList("this", "is", "a", "test"));	    
Assert.assertEquals(Math.log((1.0 / 8.0) * (1.0 / 8.0) * (1.0 / 8.0)* (1.0 / 8.0)), model4.probability(Arrays.asList("787", "asd", "dwew", "5656")), 1e-10);
}

@Test(timeout = 10000)
public void testString4gramLogprobLaplace() {
NgramLanguageModel<String> model =
new NgramLanguageModel<>(4, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model.train(Arrays.asList("1", "0", "0", "1", "1", "1", "0", "0", "1", "0", "1", "0", "1", "0", "0", "0"));
Assert.assertEquals(
Math.log((8.0 / 18.0) * (6.0 / 9.0) * (4.0 / 7.0) * (2.0 / 5.0) * (1.0 / 3.0) * (1.0 / 3.0)
* (1.0 / 3.0)),
model.probability(Arrays.asList("1", "0", "0", "0", "0", "0", "1")),
1e-10);
} 

@Test(timeout = 10000)

public void testCharacter3gramLaplace() {
NgramLanguageModel<Character> model =
new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.LAPLACE);
model.train(charactersOf("babbaaaa"));
Assert.assertEquals((6.0 / 10.0) * (4.0 / 7.0) * (3.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0)
* (2.0 / 3.0), model.probability(charactersOf("aaaabb")), 1e-10);
}

@Test(timeout = 10000)

public void testCharacter3gramAll2nd(){
NgramLanguageModel<Character> model =
new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.NONE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(((4.0 / 12.0) * (2.0 / 4.0) * (0.0 / 2.0) * (2.0 / 2.0) * (0.0 / 2.0) * (0.0 / 0.0) * (2.0 / 2.0)), model.probability(charactersOf("bcbacba")), 1e-10);

model = new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.NONE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(Math.log(((4.0 / 12.0) * (2.0 / 4.0) * (0.0 / 2.0) * (2.0 / 2.0) * (0.0 / 2.0) * (0.0 / 0.0) * (2.0 / 2.0))), model.probability(charactersOf("bcbacba")), 1e-10);

model = new NgramLanguageModel<>(3, Representation.PROBABILITY, Smoothing.LAPLACE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(((5.0 / 15.0 ) * (3.0 / 7.0 ) * (1.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0) * (1.0 / 3.0) * (3.0 / 5.0)), model.probability(charactersOf("bcbacba")), 1e-10);

model = new NgramLanguageModel<>(3, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model.train(charactersOf("abccbaabccba"));
Assert.assertEquals(Math.log((5.0 / 15.0 ) * (3.0 / 7.0 ) * (1.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0) * (1.0 / 3.0) * (3.0 / 5.0)), model.probability(charactersOf("bcbacba")), 1e-10);
}

@Test(timeout = 10000)

public void testCharacter2gram2nd() {
NgramLanguageModel<Character> model =
new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE);
model.train(charactersOf("babbaaaa"));
Assert.assertEquals((5.0 / 8.0) * (3.0 / 5.0) * (3.0 / 5.0) * (3.0 / 5.0) * (1.0 / 5.0)
* (1.0 / 3.0), model.probability(charactersOf("aaaabb")), 1e-10);
}
@Test
public void testCharacter2gramLaplace2nd() {
NgramLanguageModel<Character> model =
new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.LAPLACE);
model.train(charactersOf("babbaaaa"));
Assert.assertEquals((6.0 / 10.0) * (4.0 / 7.0) * (4.0 / 7.0) * (4.0 / 7.0) * (2.0 / 7.0)
* (2.0 / 5.0), model.probability(charactersOf("aaaabb")), 1e-10);
}

@Test(timeout = 10000)

public void testInteger4gramLogprobLaplace2nd() {
NgramLanguageModel<Integer> model =
new NgramLanguageModel<>(4, Representation.LOG_PROBABILITY, Smoothing.LAPLACE);
model.train(Arrays.asList(1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0));
Assert.assertEquals(
Math.log((8.0 / 18.0) * (6.0 / 9.0) * (4.0 / 7.0) * (2.0 / 5.0) * (1.0 / 3.0) * (1.0 / 3.0)
* (1.0 / 3.0)),
model.probability(Arrays.asList(1, 0, 0, 0, 0, 0, 1)),
1e-10);
}

@Test(timeout = 10000)
public void testLongString100gram() {
  NgramLanguageModel<Character> model =
      new NgramLanguageModel<>(100, Representation.PROBABILITY, Smoothing.NONE);
  String s = "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += "abbabbbbaaaabaababaabbaaabbbaaaabbbbabbbaacccdceccccdccccdcccccecccdcedddedddeddeedaddaeb";
  s += s + s + s + s + s + s + s + s;
  System.out.println(s.length());
  model.train(charactersOf(s));
  model.probability(charactersOf("baabbbaababbaaaaaabbbaa"));
}
@Test(timeout = 10000) public void testString2gram() 
{ 
	NgramLanguageModel<String> model = new NgramLanguageModel<>(2, Representation.PROBABILITY, Smoothing.NONE); 
	model.train(Lists.newArrayList(("A quick brown fox jumps over the lazy dog".split(" ")))); 
	model.train(Lists.newArrayList(("Any fox".split(" ")))); 
	//model.train(Lists.newArrayList(("He ran over the lazy fox".split(" ")))); 
	Assert.assertEquals((2.0 / 11.0) * (1.0 / 2.0), model.probability(Lists.newArrayList("fox jumps".split(" "))), 1e-10); 
	Assert.assertEquals((1.0 / 11.0) * (1.0 / 1.0), model.probability(Lists.newArrayList("Any fox".split(" "))), 1e-10); 
	}

}
