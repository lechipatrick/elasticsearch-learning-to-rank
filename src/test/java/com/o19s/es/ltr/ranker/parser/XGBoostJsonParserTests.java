/*
 * Copyright [2017] Wikimedia Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.o19s.es.ltr.ranker.parser;

import com.o19s.es.ltr.LtrTestUtils;
import com.o19s.es.ltr.feature.FeatureSet;
import com.o19s.es.ltr.feature.store.StoredFeature;
import com.o19s.es.ltr.feature.store.StoredFeatureSet;
import com.o19s.es.ltr.ranker.DenseFeatureVector;
import com.o19s.es.ltr.ranker.LtrRanker.FeatureVector;
import com.o19s.es.ltr.ranker.dectree.NaiveAdditiveDecisionTree;
import com.o19s.es.ltr.ranker.linear.LinearRankerTests;
import org.apache.lucene.util.LuceneTestCase;
import org.elasticsearch.common.ParsingException;
import org.elasticsearch.core.internal.io.Streams;
import org.hamcrest.CoreMatchers;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.o19s.es.ltr.LtrTestUtils.randomFeature;
import static com.o19s.es.ltr.LtrTestUtils.randomFeatureSet;
import static java.util.Collections.singletonList;

public class XGBoostJsonParserTests extends LuceneTestCase {
    private final XGBoostJsonParser parser = new XGBoostJsonParser();
    public void testReadLeaf() throws IOException {
        String model = "[ {\"nodeid\": 0, \"leaf\": 0.234}]";
        FeatureSet set = randomFeatureSet();
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        assertEquals(0.234F, tree.score(tree.newFeatureVector(null)), Math.ulp(0.234F));
    }

    public void testShiptLTR() throws IOException {
        String model = readModel("/models/SpS_xgb_model_elastic_search.json");

        // build feature names
        String feature_names[] = new String[] {
                "position_1",
                "position_2",
                "position_3",
                "position_4",
                "position_5",
                "position_6",
                "position_7",
                "position_8",
                "position_9",
                "position_10",
                "position_11",
                "position_12",
                "position_13",
                "position_14",
                "position_15",
                "position_16",
                "position_17",
                "position_18",
                "position_19",
                "position_20",
                "position_21",
                "position_22",
                "position_23",
                "position_24",
                "position_25",
                "position_26",
                "position_27",
                "position_28",
                "position_29",
                "position_30",
                "position_31",
                "position_32",
                "position_33",
                "position_34",
                "position_35",
                "position_36",
                "position_37",
                "position_38",
                "position_39",
                "position_40",
                "position_41",
                "position_42",
                "position_43",
                "position_44",
                "position_45",
                "position_46",
                "position_47",
                "position_48",
                "position_49",
                "position_50",
                "position_51",
                "num_query_tokens",
                "textual",
                "es_log_sale_rank",
                "es_product_price",
                "synonymn",
                "bm25_name",
                "bm25_brand",
                "bm25_category",
                "num_tokens_product_name",
                "query_category_feature_H1_Score",
                "query_category_feature_H2_Score",
                "query_category_feature_leafnode_Score",
                "jaccard_score_title",
                "jaccard_score_brand",
                "jaccard_score_category",
                "product_atc_rate",
                "query_product_atc_rate",
                "product_days_in_catalog",
                "log_product_days_in_catalog",
                "brand_match"};
        List<StoredFeature> features = new ArrayList<StoredFeature>();
        for (String feature_name: feature_names) {
            features.add(randomFeature(feature_name));
        }
        FeatureSet set = new StoredFeatureSet("set", features);
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        System.out.println("loaded model successfully");

        // build feature values
        float feature_values[] = new float[] {
                1F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                0F,
                1.0F,
                89.410645F,
                14.727233F,
                6.09F,
                6.884809F,
                4.6122932F,
                Float.NaN,
                5.660995F,
                8.0F,
                0.8709F,
                0.8566F,
                0.7817F,
                0.13333334F,
                0.0F,
                1.0F,
                228.0F,
                76.0F,
                2882.0F,
                7.9665866F,
                0.0F
        };
        FeatureVector v = tree.newFeatureVector(null);
        for (int i = 0; i < feature_values.length; i ++) {
            v.setFeatureScore(i, feature_values[i]);
        }

        float score = tree.score(v);
        System.out.println("raw score is: " + score);
        float normalized_score = (float) (1 / (1 + Math.exp(-score)));
        System.out.println("transformed score is: " + normalized_score);
        assertEquals(0.043071, normalized_score, 0.0001F);
    }

    public void testReadSimpleSplit() throws IOException {
        String model = "[{" +
                "\"nodeid\": 0," +
                "\"split\":\"feat1\"," +
                "\"depth\":0," +
                "\"split_condition\":0.123," +
                "\"yes\":1," +
                "\"no\": 2," +
                "\"missing\":1,"+
                "\"children\": [" +
                "   {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "   {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]";

        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        FeatureVector v = tree.newFeatureVector(null);
        v.setFeatureScore(0, 0.124F);
        assertEquals(0.2F, tree.score(v), Math.ulp(0.2F));
        v.setFeatureScore(0, 0.122F);
        assertEquals(0.5F, tree.score(v), Math.ulp(0.5F));
        v.setFeatureScore(0, 0.123F);
        assertEquals(0.2F, tree.score(v), Math.ulp(0.2F));
        v.setFeatureScore(0, Float.NaN);
        assertEquals(0.5F, tree.score(v), Math.ulp(0.2F));
    }

    public void testReadSimpleSplitInObject() throws IOException {
        String model = "{" +
                "\"splits\": [{" +
                "   \"nodeid\": 0," +
                "   \"split\":\"feat1\"," +
                "   \"depth\":0," +
                "   \"split_condition\":0.123," +
                "   \"yes\":1," +
                "   \"no\": 2," +
                "   \"missing\":2,"+
                "   \"children\": [" +
                "      {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "      {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]}";

        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        FeatureVector v = tree.newFeatureVector(null);
        v.setFeatureScore(0, 0.124F);
        assertEquals(0.2F, tree.score(v), Math.ulp(0.2F));
        v.setFeatureScore(0, 0.122F);
        assertEquals(0.5F, tree.score(v), Math.ulp(0.5F));
        v.setFeatureScore(0, 0.123F);
        assertEquals(0.2F, tree.score(v), Math.ulp(0.2F));
    }

    public void testReadSimpleSplitWithObjective() throws IOException {
        String model = "{" +
                "\"objective\": \"reg:linear\"," +
                "\"splits\": [{" +
                "   \"nodeid\": 0," +
                "   \"split\":\"feat1\"," +
                "   \"depth\":0," +
                "   \"split_condition\":0.123," +
                "   \"yes\":1," +
                "   \"no\": 2," +
                "   \"missing\":2,"+
                "   \"children\": [" +
                "      {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "      {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]}";

        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        FeatureVector v = tree.newFeatureVector(null);
        v.setFeatureScore(0, 0.124F);
        assertEquals(0.2F, tree.score(v), Math.ulp(0.2F));
        v.setFeatureScore(0, 0.122F);
        assertEquals(0.5F, tree.score(v), Math.ulp(0.5F));
        v.setFeatureScore(0, 0.123F);
        assertEquals(0.2F, tree.score(v), Math.ulp(0.2F));
    }

    public void testReadSplitWithUnknownParams() throws IOException {
        String model = "{" +
                "\"not_param\": \"value\"," +
                "\"splits\": [{" +
                "   \"nodeid\": 0," +
                "   \"split\":\"feat1\"," +
                "   \"depth\":0," +
                "   \"split_condition\":0.123," +
                "   \"yes\":1," +
                "   \"no\": 2," +
                "   \"missing\":2,"+
                "   \"children\": [" +
                "      {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "      {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]}";

        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        assertThat(expectThrows(ParsingException.class, () -> parser.parse(set, model)).getMessage(),
                CoreMatchers.containsString("Unable to parse XGBoost object"));
    }

    public void testBadObjectiveParam() throws IOException {
        String model = "{" +
                "\"objective\": \"reg:invalid\"," +
                "\"splits\": [{" +
                "   \"nodeid\": 0," +
                "   \"split\":\"feat1\"," +
                "   \"depth\":0," +
                "   \"split_condition\":0.123," +
                "   \"yes\":1," +
                "   \"no\": 2," +
                "   \"missing\":2,"+
                "   \"children\": [" +
                "      {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "      {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]}";

        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        assertThat(expectThrows(ParsingException.class, () -> parser.parse(set, model)).getMessage(),
                CoreMatchers.containsString("Unable to parse XGBoost object"));
    }

    public void testReadWithLogisticObjective() throws IOException {
        String model = "{" +
                "\"objective\": \"reg:logistic\"," +
                "\"splits\": [{" +
                "   \"nodeid\": 0," +
                "   \"split\":\"feat1\"," +
                "   \"depth\":0," +
                "   \"split_condition\":0.123," +
                "   \"yes\":1," +
                "   \"no\": 2," +
                "   \"missing\":2,"+
                "   \"children\": [" +
                "      {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "      {\"nodeid\": 2, \"depth\": 1, \"leaf\": -0.2}" +
                "]}]}";

        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        FeatureVector v = tree.newFeatureVector(null);
        v.setFeatureScore(0, 0.124F);
        assertEquals(0.45016602F, tree.score(v), Math.ulp(0.45016602F));
        v.setFeatureScore(0, 0.122F);
        assertEquals(0.62245935F, tree.score(v), Math.ulp(0.62245935F));
        v.setFeatureScore(0, 0.123F);
        assertEquals(0.45016602F, tree.score(v), Math.ulp(0.45016602F));
    }

    public void testMissingField() throws IOException {
        String model = "[{" +
                "\"nodeid\": 0," +
                "\"split\":\"feat1\"," +
                "\"depth\":0," +
                "\"split_condition\":0.123," +
                "\"no\": 2," +
                "\"missing\":2,"+
                "\"children\": [" +
                "   {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "   {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]";
        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        assertThat(expectThrows(ParsingException.class, () -> parser.parse(set, model)).getMessage(),
                CoreMatchers.containsString("This split does not have all the required fields"));
    }

    public void testBadStruct() throws IOException {
        String model = "[{" +
                "\"nodeid\": 0," +
                "\"split\":\"feat1\"," +
                "\"depth\":0," +
                "\"split_condition\":0.123," +
                "\"yes\":1," +
                "\"no\": 3," +
                "\"children\": [" +
                "   {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "   {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]";
        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        assertThat(expectThrows(ParsingException.class, () -> parser.parse(set, model)).getMessage(),
                CoreMatchers.containsString("Split structure is invalid, yes, no and/or"));
    }

    public void testMissingFeat() throws IOException {
        String model = "[{" +
                "\"nodeid\": 0," +
                "\"split\":\"feat2\"," +
                "\"depth\":0," +
                "\"split_condition\":0.123," +
                "\"yes\":1," +
                "\"no\": 2," +
                "\"missing\":2,"+
                "\"children\": [" +
                "   {\"nodeid\": 1, \"depth\": 1, \"leaf\": 0.5}," +
                "   {\"nodeid\": 2, \"depth\": 1, \"leaf\": 0.2}" +
                "]}]";
        FeatureSet set = new StoredFeatureSet("set", singletonList(randomFeature("feat1")));
        assertThat(expectThrows(ParsingException.class, () -> parser.parse(set, model)).getMessage(),
                CoreMatchers.containsString("Unknown feature [feat2]"));
    }

    public void testComplexModel() throws Exception {
        String model = readModel("/models/xgboost-wmf.json");
        List<StoredFeature> features = new ArrayList<>();
        List<String> names = Arrays.asList("all_near_match",
                "category",
                "heading",
                "incoming_links",
                "popularity_score",
                "redirect_or_suggest_dismax",
                "text_or_opening_text_dismax",
                "title");
        for (String n : names) {
            features.add(LtrTestUtils.randomFeature(n));
        }

        StoredFeatureSet set = new StoredFeatureSet("set", features);
        NaiveAdditiveDecisionTree tree = parser.parse(set, model);
        DenseFeatureVector v = tree.newFeatureVector(null);
        assertEquals(v.scores.length, features.size());

        for (int i = random().nextInt(5000) + 1000; i > 0; i--) {
            LinearRankerTests.fillRandomWeights(v.scores);
            assertFalse(Float.isNaN(tree.score(v)));
        }
    }

    private String readModel(String model) throws IOException {
        try (InputStream is = this.getClass().getResourceAsStream(model)) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            Streams.copy(is,  bos);
            return bos.toString(StandardCharsets.UTF_8.name());
        }
    }
}
