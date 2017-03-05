$(document).ready(function() {
    google.charts.load('current', {'packages':['line', 'corechart', 'scatter']});
    //google.charts.setOnLoadCallback(drawChart);
    $("#pred_chart").hide();
    $("#results").hide();
    $("#company_values_chart_div").hide();
    $("#method_values_chart_div").hide();
    $("#pred_values_chart_div").hide();
    createWidgets();
    var m_newData = false;



    $('#reset').on('click', function () {
        var data = {};
        data['operation'] = 'reset';
        jQuery.ajax({   type: "POST",
                        data: data,
                        success: function(data) {
                            var obj = JSON.parse(data);
                            location.reload(); 
                        },
                    });

    });

    $('#select_symbol').on('change', function () {
        var data = {};
        data['operation'] = 'update_values_chart';
        data['symbol'] = $("#select_symbol").jqxDropDownList('val');
        if (m_newData == true){
            data['period'] = $('#select_companyPeriod').jqxNumberInput('getDecimal');
        }
        $('#companyLoader').jqxLoader('open');
        jQuery.ajax({   type: "POST",
                        data: data,
                        success: function(data) {
                            var obj = JSON.parse(data);
                            $('#companyLoader').jqxLoader('close');
                            companyValuesChart(obj);
                            companyValuesStr(obj['vStr']); 
                        },
                    });

    });

    $('#select_method').on('change', function () {
        var data = {};
        data['operation'] = 'set_method';
        data['symbol'] = $("#select_symbol").jqxDropDownList('val');
        data['method'] = $("#select_method").jqxDropDownList('val');
        $('#methodLoader').jqxLoader('open');
        jQuery.ajax({   type: "POST",
                        data: data,
                        success: function(data) {
                            var obj = JSON.parse(data);
                            $('#methodLoader').jqxLoader('close');
                            methodValuesChart(obj);
                            methodValuesStr(obj['vStr']); 
                        },
                    });

    });

    $('#select_period').on('change', function () {
        var data = {};
        data['operation'] = 'new_pred';
        data['period'] = $("#select_period").jqxDropDownList('val');
        jQuery.ajax({   type: "POST",
                        data: data,
                        success: function(data) {
                            var obj = JSON.parse(data);
                            predictionValuesChart(obj);
                        },
                    });

    });

    $("#new_data").bind('change', function (event) {
        $("#new_data_div").show();
        $("#select_data_div").show();
        $("#new_data").hide();
        $("#old_data").hide();
        m_newData = true;
    });

    $("#old_data").bind('change', function (event) {
        $("#select_data_div").show();
        $("#new_data").hide();
        $("#old_data").hide();
        m_newData = false;
    });

});


function createWidgets() {

    $("#select_period_div").hide();
    $("#select_method_div").hide();
    $("#new_data_div").hide();
    $("#select_data_div").hide();

    $("#methodLoader").jqxLoader({ width: 250, height: 150, autoOpen: true });
    $('#methodLoader').jqxLoader('close');
    $("#companyLoader").jqxLoader({ width: 250, height: 150, autoOpen: true });
    $('#companyLoader').jqxLoader('close');

//    $('#select_symbol').jqxTextArea({ width: 250, height: 100, placeHolder: 'Enter a sentence...' });

    $("#new_data").jqxRadioButton({ width: 200, height: 25 });
    $("#old_data").jqxRadioButton({ width: 200, height: 25 });

    var period_source = [
        "1 day",
        "5 days",
        "10 days",
        "15 days",
        "30 days",
        "45 days"
    ];

    $("#select_period").jqxDropDownList({ source: period_source, 
                                    width: '200', 
                                    height: '25'});


    var method_source = [
        "Support Vector Machine",
        "Ensemble",
        "Gaussian",
        "K-Nearest Neighbor",
        "Neural Network",
        "Deep Learning"
    ];

    $("#select_method").jqxDropDownList({ source: method_source, 
                                    width: '200', 
                                    height: '25'});
/*    var file = "file:file.txt"
//    var file = ["company_symbol.txt"]
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function (){
        if(rawFile.readyState === 4) {
            if(rawFile.status === 200 || rawFile.status == 0){
                var allText = rawFile.responseText;i
                var symbol_source = allText.split(',');
                //alert(allText);
            }
        }
    }
    rawFile.send(null);

//    var symbol_source = ["company_symbol.txt"]*/

    $("#select_companyPeriod").jqxNumberInput({ 
        width: '100px', 
        height: '25px',
        decimalDigits: 0,
        digits: 2,
        spinButtons: true,
        decimal: 1,
        min: 1,
        max: 50,
        symbol: ' Year(s)',
        symbolPosition: 'right',
        inputMode: 'simple',
        theme: 'energyblue' 
    });

    var symbol_source = [
        "YHOO",
        "AAPL",
        "BBDO",
        "BSBR",
        "PBR",
        "BUD",
        "NVS",
        "HSBC",
        "GOOGL",
        "MSFT",
        "AMZN",
        "FB",
        "ABEV",
        "ERJ"
    ];
    $("#select_symbol").jqxDropDownList({ source: symbol_source, 
                                    //selectedIndex: 1, 
                                    width: '200', 
                                    height: '25'});

    $("#reset").jqxButton({
        width: '150',
        height: '25'
    });

}

function companyValuesStr(data) {
    $("#company_values_str").jqxListBox({ source: data, width: '900px', height: '100px',});
}

function methodValuesStr(data) {
    $("#method_values_str").jqxListBox({ source: data, width: '900px', height: '100px',});
}

function companyValuesChart(dt) {

    var data = new google.visualization.DataTable();
    data.addColumn('date', 'Day');
    data.addColumn('number', "Adjustment Price [$]");
    data.addColumn('number', "Open Price [$]");
    data.addColumn('number', "Close Price [$]");

    for (i=0; i < dt['date'].length; i++) {
        data.addRows([[new Date(dt['date'][i].split('-')), 
                        dt['adj_price'][i], 
                        dt['open_price'][i],
                        dt['close_price'][i]
        
        ]]);
    }

    var cTitle = dt['company_name'] + ' Results';
    var materialOptions = {
        chart : {title: cTitle},
        curveType: 'function',
        //hAxis: { title: 'Time'},
        //vAxis: { title: '$'},
        width: 900,
        height: 500,
        backgroundColor: '#f1f8e9',
        explorer: { actions: ['dragToZoom', 'rightClickToReset'] },
        legend: { position: 'bottom' }
    };
   
    var chartDiv = document.getElementById('company_values_chart'); 
    var materialChart = new google.charts.Line(chartDiv);
    materialChart.draw(data, materialOptions);

    $("#company_values_chart_div").show();
    $("#select_method_div").show();
}

function methodValuesChart(dt) {

    var data = new google.visualization.DataTable();
    data.addColumn('date', 'Day');
    data.addColumn('number', "Adjustment Price [$]");
    data.addColumn('number', "Predicted Price [$]");

    for (i=0; i < dt['date'].length; i++) {
        data.addRows([[new Date(dt['date'][i].split('-')), 
                        dt['adj_price'][i], 
                        dt['pred_values'][i]
        
        ]]);
    }

    var cTitle = dt['company_name'] + ' Prediction';
    var materialOptions = {
        chart : {title: cTitle},
        curveType: 'function',
        width: 900,
        height: 500,
        series: {
            0:{color: 'black', visibleInLegend: true, lineWidth: 0, pointSize: 2, 
                pointsVisible: true},
            1:{color: 'red', visibleInLegend: true, lineWidth: 2, pointSize: 0, 
                pointsVisible: false }
        },
        backgroundColor: '#f1f8e9',
        explorer: { actions: ['dragToZoom', 'rightClickToReset'], axis: 'horizontal' },
        legend: { position: 'bottom' }
    };
   
    var chartDiv = document.getElementById('method_values_chart'); 
    var materialChart = new google.charts.Scatter(chartDiv);
    //var materialChart = new google.visualization.ScatterChart(chartDiv);
    materialChart.draw(data, materialOptions);

    $("#method_values_chart_div").show();
    $("#select_period_div").show();
}

function predictionValuesChart(dt) {

    var data = new google.visualization.DataTable();
    data.addColumn('date', 'Day');
    data.addColumn('number', "Adjustment Predicted Price [$]");

    for (i=0; i < dt['date'].length; i++) {
        data.addRows([[new Date(dt['date'][i].split('-')), 
                        dt['pred_values'][i], 
        ]]);
    }

    var cTitle = dt['company_name'] + ' Predicted Prices';
    var materialOptions = {
        chart : {title: cTitle},
        curveType: 'function',
        width: 900,
        height: 500,
        series: {
            0:{color: 'green', visibleInLegend: true, lineWidth: 0, pointSize: 2, 
                pointsVisible: true}
        },
        backgroundColor: '#f1f8e9',
        explorer: { actions: ['dragToZoom', 'rightClickToReset'] },
        legend: { position: 'bottom' }
    };
   
    var chartDiv = document.getElementById('pred_values_chart'); 
    var materialChart = new google.charts.Scatter(chartDiv);
    materialChart.draw(data, materialOptions);

    $("#pred_values_chart_div").show();
}
