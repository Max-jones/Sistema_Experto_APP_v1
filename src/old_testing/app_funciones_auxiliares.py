
def show_cv_iterations(n_splits, X, y, timeseries=True):
    # https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4
    if timeseries:
        cv = TimeSeriesSplit(n_splits)
    else:
        cv = KFold(n_splits)
    
    figure, ax = plt.subplots(figsize=(10, 5))

    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        
        p1 = ax.scatter(tr, [ii] * len(tr), c='black', marker="_", lw=8)
        p2 = ax.scatter(tt, [ii] * len(tt), c='red', marker="_", lw=8)
        ax.set(
            title="Behavior of TimeseriesSplit",
            xlabel="Data Index",
            ylabel="CV Iteration",
            ylim=[5, -1],
        )
        ax.legend([p1, p2], ["Training", "Validation"])
    st.pyplot(fig=figure)
    return cv
