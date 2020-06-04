# run me if you like to run things

for output_dir in ihs_lei/**/*; do
    echo "Creating __init__ file in ${output_dir}"
    output_file="${output_dir}/__init__.py"
    echo "# IHS Markit Leading Economic Indicators\n# Developed by Eduardo Sahione" > $output_file
done

# let's see what we just did
grep "" ihs_lei/**/*.py