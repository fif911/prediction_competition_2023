from viewser import Queryset, Column

# read in country list with country names for presentation purposes
# qs = (Queryset("country_list", "country_month")
#
#       .with_column(Column("id", from_table="country", from_column="id"))
#       .with_column(Column("name", from_table="country", from_column="name"))
#
#       )
# countrylist = qs.publish().fetch().loc[504]
new_queryset = (Queryset("simple_conflict", "country_month")

           .with_column(Column("ged_sb", from_loa="country_month", from_column="ged_sb_best_sum_nokgi")
                        .aggregate('avg')
                        .transform.ops.ln()
                        .transform.missing.replace_na()
                       )
               )
data = new_queryset.publish().fetch()
print(data)