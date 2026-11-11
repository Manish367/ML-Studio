const ALGO = {
  "Linear Regression": {
    how: "Fits the straight line (or flat plane) that gets closest to all the points by minimising squared errors.",
    best: "Simple relationships and a fast, easy-to-explain baseline.",
    watch: "Cannot bend to curves, and a few extreme values can drag the line.",
  },
  "Polynomial Regression": {
    how: "Adds squared and higher-power versions of each column, then fits a linear model, which lets the line curve.",
    best: "Smooth curved relationships on small to medium data.",
    watch:
      "High degrees overfit quickly and behave wildly outside the training range.",
  },
  Ridge: {
    how: "Linear regression with a penalty on large coefficients, which keeps every column's weight small and stable.",
    best: "Many columns that overlap or are correlated.",
    watch: "Still linear. Strong penalties make the model too simple.",
  },
  Lasso: {
    how: "Linear regression whose penalty pushes the weights of unhelpful columns all the way to zero.",
    best: "Picking the few columns that matter from many.",
    watch: "Among correlated columns it keeps one more or less at random.",
  },
  "Decision Tree": {
    how: "Asks a series of yes/no questions about the columns, splitting the data into smaller and purer groups.",
    best: "Rules you can read, mixed column types and non-linear patterns.",
    watch:
      "A deep tree memorises the training data. Small changes in data change the tree.",
  },
  "Random Forest": {
    how: "Grows many different trees on random samples of rows and columns, then averages their answers.",
    best: "A strong, low-fuss default for tables of data.",
    watch: "Slower and bigger than one tree, and less easy to interpret.",
  },
  KNN: {
    how: "Predicts from the K most similar training rows, taking their average (numbers) or majority vote (classes).",
    best: "Small datasets with a few well-scaled columns.",
    watch: "Slow on big data and hurt by irrelevant or unscaled columns.",
  },
  SVR: {
    how: "Fits a smooth curve that stays within a tolerance band around the data and ignores points inside the band.",
    best: "Medium-sized data with smooth, non-linear patterns.",
    watch:
      "Slow beyond about 10,000 rows and sensitive to the C and gamma settings.",
  },
  "Gradient Boosting": {
    how: "Builds small trees one after another, each correcting the mistakes of the ones before.",
    best: "Often the most accurate choice on tabular data.",
    watch: "Sensitive to settings, can overfit, and trains one step at a time.",
  },
  "Logistic Regression": {
    how: "Weighs each column, then squashes the total into a probability between 0 and 1 for each class.",
    best: "A fast, explainable baseline, and well-calibrated probabilities.",
    watch:
      "Draws straight boundaries only, so tangled classes need other models.",
  },
  SVM: {
    how: "Finds the boundary with the widest possible gap between classes, and can bend it using kernels.",
    best: "Medium-sized data with clear but curved class boundaries.",
    watch: "Slow on large data and needs scaled columns.",
  },
  "Naive Bayes": {
    how: "Uses Bayes' rule, treating every column as independent evidence for each class.",
    best: "Very fast training, small data and quick baselines.",
    watch:
      "The independence assumption is often wrong, so probabilities are over-confident.",
  },
  AdaBoost: {
    how: "Trains many tiny trees in turn, giving more weight to rows the earlier ones got wrong.",
    best: "Moderately sized, fairly clean data.",
    watch: "Sensitive to noisy rows and outliers.",
  },
  "Extra Trees": {
    how: "Like a random forest, but the split points are chosen at random, which makes the trees even more varied.",
    best: "Fast, strong results on noisy tabular data.",
    watch: "Less interpretable and can be bulky in memory.",
  },
  kmeans: {
    how: "Places K centres, assigns each row to its nearest centre, moves the centres to the group averages and repeats.",
    best: "Round, similar-sized groups.",
    watch: "You must choose K, and it struggles with odd shapes and outliers.",
  },
  dbscan: {
    how: "Grows groups from dense areas of points and labels sparse points as noise.",
    best: "Odd-shaped groups and spotting outliers without choosing K.",
    watch:
      "The neighbourhood size (eps) is touchy, and groups of different density are hard.",
  },
  agglomerative: {
    how: "Starts with every row alone and repeatedly merges the two closest groups, building a tree you can cut at any height.",
    best: "Smaller datasets where the tree itself is interesting.",
    watch: "Slow and memory-hungry on big data, so this app uses a sample.",
  },
  gmm: {
    how: "Models the data as a mix of bell-shaped clusters and gives each row a probability of belonging to each.",
    best: "Overlapping, oval-shaped groups.",
    watch: "Needs K, and can get stuck in a poor fit.",
  },
  meanshift: {
    how: "Slides a window uphill towards the densest spots and treats each peak as a cluster centre.",
    best: "When you do not know K and the groups are blob-like.",
    watch: "Slow on large data and the window size matters a lot.",
  },
  isolation_forest: {
    how: "Randomly cuts the data. Outliers get isolated in just a few cuts, so quick isolation means likely outlier.",
    best: "Fast anomaly detection on many columns.",
    watch: "You must guess the expected share of outliers.",
  },
  lof: {
    how: "Compares each point's local density with its neighbours'. Points in much emptier spots are outliers.",
    best: "Outliers that are only odd relative to their own neighbourhood.",
    watch: "Slow on big data and sensitive to the neighbourhood size.",
  },
  oneclass_svm: {
    how: "Learns a boundary around what normal data looks like, and flags anything outside it.",
    best: "Small, mostly-normal data with a clean boundary.",
    watch: "Slow beyond a few thousand rows and sensitive to settings.",
  },
};
export const AlgoNote = ({ name }) => {
  const a = ALGO[name];
  if (!a) return null;
  return (
    <details className="small mt-2">
      <summary style={{ cursor: "pointer", color: "#22d3ee" }}>
        How does {name} work?
      </summary>
      <div className="mt-1 opacity-75">
        <b>How it works:</b> {a.how}
        <br />
        <b>Best for:</b> {a.best}
        <br />
        <b>Watch out:</b> {a.watch}
      </div>
    </details>
  );
};
